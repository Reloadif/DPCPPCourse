#include <sycl/sycl.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <exception>
#include <numeric>
#include <cmath>

const int BlockSize = 16;

const double XStart = 0;
const double XEnd = 1;

const double YStart = 0;
const double YEnd = 1;

const double ExpectedResult = 0.386822;

double function(double x, double y) {
	return sin(x) * cos(y);
}

int main(int argc, char* argv[]) {
	if (argc > 4) {
		std::cout << "Invalid arguments. Two arguments expected!" << std::endl;
		return 0;
	}

	int intervalsPerDimension = 1000;
	if (argc > 1) {
		try {
			intervalsPerDimension = std::stoi(argv[1]);
		}
		catch (const std::exception& e) {
			std::cout << "Invalid arguments. Conversion error!" << std::endl;
			std::terminate();
		}
	}

	std::string selectedDevice = "cpu";
	if (argc > 2) {
		if (!(std::string(argv[2]).compare("cpu") == 0 || std::string(argv[2]).compare("gpu") == 0)) {
			std::cout << "Invalid arguments. Unexpected second argument! (Use 'cpu' or 'gpu')" << std::endl;
			return 0;
		}

		selectedDevice = argv[2];
	}

	sycl::queue mainQueue(selectedDevice.compare("cpu") == 0 ? sycl::cpu_selector_v : sycl::gpu_selector_v, { sycl::property::queue::enable_profiling() });

	int numberOfGroups = ceil(static_cast<double>(intervalsPerDimension) / BlockSize);
	int iterationCount = BlockSize * numberOfGroups;

	sycl::event mainEvent;
	std::vector<double> groupResult(numberOfGroups * numberOfGroups, .0);
	{
		sycl::buffer buffGroupResult(groupResult);
		mainEvent = mainQueue.submit([&](sycl::handler& cgh) {
			sycl::accessor AccGroupResult(buffGroupResult, cgh, sycl::write_only);
			double dx = static_cast<double>(XEnd - XStart) / iterationCount;
			double dy = static_cast<double>(YEnd - YStart) / iterationCount;

			cgh.parallel_for(
				sycl::nd_range<2>(sycl::range<2>(iterationCount, iterationCount), sycl::range<2>(BlockSize, BlockSize)), [=](sycl::nd_item<2> item) {
					double x = XStart + static_cast<double>(item.get_global_id(0)) / iterationCount * (XEnd - XStart);
					double y = YStart + static_cast<double>(item.get_global_id(1)) / iterationCount * (YEnd - YStart);
					double df = function(x + dx / 2, y + dy / 2) * dx * dy;

					double result = sycl::reduce_over_group(item.get_group(), df, std::plus<double>());
					if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
						AccGroupResult[item.get_group(0) * item.get_group_range(0) + item.get_group(1)] = result;
					}
				});
			});
		mainQueue.wait();
	}

	auto start = mainEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
	auto end = mainEvent.get_profiling_info<sycl::info::event_profiling::command_end>();

	double result = std::reduce(groupResult.cbegin(), groupResult.cend(), .0);

	std::cout << "Kernel Execution Time: " << (end - start) / 10e6 << " ms" << std::endl;
	std::cout << "Expected: " << ExpectedResult << std::endl;
	std::cout << "Computed: " << result << std::endl;
	std::cout << "Difference: " << abs(ExpectedResult - result) << std::endl;

	return 0;
}