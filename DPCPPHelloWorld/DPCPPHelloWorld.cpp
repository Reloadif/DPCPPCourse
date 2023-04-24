#include<CL/sycl.hpp>

#include<iostream>
#include<vector>
#include<string>

using namespace cl::sycl;

static const int N = 4;

int main(int argc, char* argv[]) {
	std::vector<std::pair<_V1::device, std::pair<size_t, size_t>>> allDevices;

	auto platforms = platform::get_platforms();
	for (size_t p = 0; p < platforms.size(); ++p)
	{
		std::cout << "Platform #" << p << ": " << platforms[p].get_info<info::platform::name>() << std::endl;
		auto devices = platforms[p].get_devices(info::device_type::all);
		for (size_t d = 0; d < devices.size(); ++d)
		{
			std::cout << "-- Device #" << d << ": " << devices[d].get_info<info::device::name>() << std::endl;
			allDevices.push_back(std::pair(devices[d], std::pair(p, d)));
		}
	}

	std::cout << std::endl;

	for (const auto& d : allDevices) {
		std::cout << d.first.get_info<info::device::name>() << std::endl;
		auto currentIds = d.second;

		queue q{ d.first };
		q.submit([&](handler& cgh) {
			sycl::stream out(1024, 256, cgh);

			cgh.parallel_for(range<1>(N), [=](id<1> i) {
				out << "[" << i.get(0) << "]" << " Hello from platform " << currentIds.first << " and device " << currentIds.second << endl;
				});
			});
		q.wait();

		std::cout << std::endl;
	}

	return 0;
}