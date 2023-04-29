#include <CL/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>

#include "CustomRandom.hpp"

static auto exception_handler = [](sycl::exception_list e_list) {
	for (std::exception_ptr const& e : e_list) {
		try {
			std::rethrow_exception(e);
		}
		catch (std::exception const& e) {
			std::cout << "Failure" << std::endl;
		}
	}
};

template <typename T> inline
T tabs(const T& v) { return v < 0 ? -v : v; }

template<typename T>
T vectorNorm(const T* vector, const int size) {
	T result = tabs(vector[0]);
	for (int i = 1; i < size; ++i) {
		if (tabs(vector[i]) > result) {
			result = tabs(vector[i]);
		}
	}

	return result;
}

template<typename T>
T achivedAccuracy(const T* A, const T* b, const T* x, const int numberEquations) {
	T* diff = new T[numberEquations];
	for (int i = 0; i < numberEquations; ++i) {
		T sum = 0;
		for (int j = 0; j < numberEquations; ++j) {
			sum += A[j * numberEquations + i] * x[j];
		}
		diff[i] = sum - b[i];
	}

	T result = vectorNorm(diff, numberEquations);
	delete[] diff;

	return result;
}

template <typename T>
void JacobiOnAccessors(const sycl::device& device, const T* A, const T* b, const int numberEquations, const double targetAccuracy, const int maximalNumberIterations) {
	sycl::queue que(device, exception_handler, { sycl::property::queue::enable_profiling() });
	size_t allTime = 0;
	int it = 0;

	T* x1 = new T[numberEquations];
	memset(x1, 0, sizeof(T) * numberEquations);
	T* x2 = new T[numberEquations];
	T* diff = new T[numberEquations];
	{
		sycl::buffer<T> bufA(A, numberEquations * numberEquations);
		sycl::buffer<T> bufB(b, numberEquations);
		sycl::buffer<T> bufX1(x1, numberEquations);
		sycl::buffer<T> bufX2(x2, numberEquations);
		sycl::buffer<T> bufDiff(diff, numberEquations);

		auto pBuffX1 = &bufX1;
		auto pBuffX2 = &bufX2;
		for (it = 0; it < maximalNumberIterations; ++it) {
			sycl::event e = que.submit([&](sycl::handler& cgh) {
				sycl::accessor accA(bufA, cgh, sycl::read_only);
				sycl::accessor accB(bufB, cgh, sycl::read_only);
				sycl::accessor accX1(*pBuffX1, cgh, sycl::read_only);
				sycl::accessor accX2(*pBuffX2, cgh, sycl::write_only);
				sycl::accessor accDiff(bufDiff, cgh, sycl::write_only);

				cgh.parallel_for(sycl::range<1>(numberEquations), [=](sycl::id<1> item) {
					accX2[item] = accB[item];
					for (int i = 0; i < numberEquations; ++i) {
						accX2[item] -= accA[i * numberEquations + item] * accX1[i];
					}

					accX2[item] += accA[item * numberEquations + item] * accX1[item];
					accX2[item] /= accA[item * numberEquations + item];
					accDiff[item] = accX1[item] - accX2[item];
					});
				});
			e.wait();

			allTime += e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>();
			std::swap(pBuffX1, pBuffX2);

			auto hostDiff = bufDiff.get_host_access();
			auto hostX1 = pBuffX1->get_host_access();

			if (vectorNorm(hostDiff.get_pointer(), numberEquations) / vectorNorm(hostX1.get_pointer(), numberEquations) < targetAccuracy) {
				break;
			}
		}
	}
	std::cout << "[Accessors] Time: " << static_cast<double>(allTime) / 10e6 << " ms Accuracy: " << achivedAccuracy(A, b, x1, numberEquations) << " Iterations: " << it << std::endl;

	delete[] x1;
	delete[] x2;
	delete[] diff;
}

template <typename T>
void JacobiOnShared(const sycl::device& device, const T* A, const T* b, const int numberEquations, const double targetAccuracy, const int maximalNumberIterations) {
	sycl::queue que(device, exception_handler, { sycl::property::queue::enable_profiling() });
	size_t allTime = 0;
	int it = 0;

	T* x = new T[numberEquations];
	{
		T* sharedA = sycl::malloc_shared<T>(numberEquations * numberEquations, que);
		T* sharedB = sycl::malloc_shared<T>(numberEquations, que);
		T* sharedX1 = sycl::malloc_shared<T>(numberEquations, que);
		T* sharedX2 = sycl::malloc_shared<T>(numberEquations, que);
		T* sharedDiff = sycl::malloc_shared<T>(numberEquations, que);

		for (int i = 0; i < numberEquations; ++i) {
			sharedB[i] = b[i];
			sharedX1[i] = 0;
			for (int j = 0; j < numberEquations; ++j) {
				sharedA[i * numberEquations + j] = A[i * numberEquations + j];
			}
		}

		for (it = 0; it < maximalNumberIterations; ++it) {
			sycl::event e = que.submit([&](sycl::handler& cgh) {
				cgh.parallel_for(sycl::range<1>(numberEquations), [=](sycl::id<1> item) {
					sharedX2[item] = sharedB[item];
					for (int i = 0; i < numberEquations; ++i) {
						sharedX2[item] -= sharedA[i * numberEquations + item] * sharedX1[i];
					}

					sharedX2[item] += sharedA[item * numberEquations + item] * sharedX1[item];
					sharedX2[item] /= sharedA[item * numberEquations + item];
					sharedDiff[item] = sharedX1[item] - sharedX2[item];
					});
				});
			e.wait();

			allTime += e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>();
			std::swap(sharedX1, sharedX2);

			if (vectorNorm(sharedDiff, numberEquations) / vectorNorm(sharedX1, numberEquations) < targetAccuracy) {
				break;
			}
		}

		for (int i = 0; i < numberEquations; ++i) {
			x[i] = sharedX1[i];
		}

		sycl::free(sharedA, que);
		sycl::free(sharedB, que);
		sycl::free(sharedX1, que);
		sycl::free(sharedX2, que);
		sycl::free(sharedDiff, que);
	}
	std::cout << "[Shared]    Time: " << static_cast<double>(allTime) / 10e6 << " ms Accuracy: " << achivedAccuracy(A, b, x, numberEquations) << " Iterations: " << it << std::endl;

	delete[] x;
}

template <typename T>
void JacobiOnDevice(const sycl::device& device, const T* A, const T* b, const int numberEquations, const double targetAccuracy, const int maximalNumberIterations) {
	sycl::queue que(device, exception_handler, { sycl::property::queue::enable_profiling() });
	size_t allTime = 0;
	int it = 0;

	T* x = new T[numberEquations];
	T* diff = new T[numberEquations];
	{
		T* deviceA = sycl::malloc_device<T>(numberEquations * numberEquations, que);
		T* deviceB = sycl::malloc_device<T>(numberEquations, que);
		T* deviceX1 = sycl::malloc_device<T>(numberEquations, que);
		T* deviceX2 = sycl::malloc_device<T>(numberEquations, que);
		T* deviceDiff = sycl::malloc_device<T>(numberEquations, que);

		que.memcpy(deviceA, A, numberEquations * numberEquations * sizeof(T)).wait();
		que.memcpy(deviceB, b, numberEquations * sizeof(T)).wait();
		que.memset(deviceX1, 0, numberEquations * sizeof(T)).wait();

		for (it = 0; it < maximalNumberIterations; ++it) {
			sycl::event e = que.submit([&](sycl::handler& cgh) {
				cgh.parallel_for(sycl::range<1>(numberEquations), [=](sycl::id<1> item) {
					deviceX2[item] = deviceB[item];
					for (int i = 0; i < numberEquations; ++i) {
						deviceX2[item] -= deviceA[i * numberEquations + item] * deviceX1[i];
					}

					deviceX2[item] += deviceA[item * numberEquations + item] * deviceX1[item];
					deviceX2[item] /= deviceA[item * numberEquations + item];
					deviceDiff[item] = deviceX1[item] - deviceX2[item];
					});
				});
			e.wait();

			allTime += e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>();
			std::swap(deviceX1, deviceX2);

			que.memcpy(diff, deviceDiff, numberEquations * sizeof(T)).wait();
			que.memcpy(x, deviceX1, numberEquations * sizeof(T)).wait();

			if (vectorNorm(diff, numberEquations) / vectorNorm(x, numberEquations) < targetAccuracy) {
				break;
			}
		}

		sycl::free(deviceA, que);
		sycl::free(deviceB, que);
		sycl::free(deviceX1, que);
		sycl::free(deviceX2, que);
		sycl::free(deviceDiff, que);
	}
	std::cout << "[Device]    Time: " << static_cast<double>(allTime) / 10e6 << " ms Accuracy: " << achivedAccuracy(A, b, x, numberEquations) << " Iterations: " << it << std::endl;

	delete[] x;
}

int main(int argc, char* argv[]) {
	if (argc > 5) {
		std::cout << "Invalid arguments. Five arguments expected!" << std::endl;
		return 0;
	}

	int numberEquations = 1000;
	if (argc > 1) {
		try {
			numberEquations = std::stoi(argv[1]);
		}
		catch (const std::exception& e) {
			std::cout << "Invalid first argument. Conversion error!" << std::endl;
			std::terminate();
		}
	}

	double targetAccuracy = 1 / 10e3;
	if (argc > 2) {
		try {
			targetAccuracy = std::stod(argv[2]);
		}
		catch (const std::exception& e) {
			std::cout << "Invalid second argument. Conversion error!" << std::endl;
			std::terminate();
		}
	}

	int maximalNumberIterations = 100;
	if (argc > 3) {
		try {
			maximalNumberIterations = std::stoi(argv[3]);
		}
		catch (const std::exception& e) {
			std::cout << "Invalid third argument. Conversion error!" << std::endl;
			std::terminate();
		}
	}

	std::string selectedDevice = "cpu";
	if (argc > 4) {
		if (!(std::string(argv[4]).compare("cpu") == 0 || std::string(argv[4]).compare("gpu") == 0)) {
			std::cout << "Unexpected fourth argument! (Use 'cpu' or 'gpu')" << std::endl;
			return 0;
		}

		selectedDevice = argv[4];
	}
	sycl::device device = selectedDevice.compare("cpu") == 0 ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v);
	std::cout << "Target device: " << device.get_info<sycl::info::device::name>() << std::endl;

	double* A = new double[numberEquations * numberEquations];
	double* b = new double[numberEquations];

	getRandomMatrix(A, numberEquations, numberEquations);
	getRandomVector(b, numberEquations); 

	JacobiOnAccessors(device, A, b, numberEquations, targetAccuracy, maximalNumberIterations);
	JacobiOnShared(device, A, b, numberEquations, targetAccuracy, maximalNumberIterations);
	JacobiOnDevice(device, A, b, numberEquations, targetAccuracy, maximalNumberIterations);

	delete[] A;
	delete[] b;

	return 0;
}