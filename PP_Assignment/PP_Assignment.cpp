#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	typedef unsigned char mytype;

	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	int nr_bins = 256;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { nr_bins = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		
		CImgDisplay disp_input;
		bool bit_16 = false;

		string f;
		fstream file;
		file.open(image_filename);
		for (int i = 0; i < 10; i++)
		{
			getline(file, f);
			cout << f << endl;
			if (f == to_string(255)) {
				disp_input = CImgDisplay(CImg<unsigned char>(image_filename.c_str()), "input");
				break;
			}
			else if (f == to_string(65535)) {
				disp_input = CImgDisplay(CImg<unsigned short>(image_filename.c_str()), "input");
				bit_16 = true;
				break;
			}
		}
		
		CImg<unsigned char> image_input(image_filename.c_str());
		CImg<unsigned short> image_input_16(image_filename.c_str());
		CImg<unsigned short> original;

		if (bit_16) {
			original = image_input_16;
		}
		else {
			original = image_input;
		}
		
		CImg<unsigned char> cb;
		CImg<unsigned char> cr;

		if (image_input.spectrum() == 3) {
			cb = original.get_RGBtoYCbCr().channel(1);
			cr = original.get_RGBtoYCbCr().channel(2);
			image_input = image_input.get_RGBtoYCbCr();
			image_input = image_input.channel(0);
		}

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int tvector;

		//Part 3 - memory allocation
		//host - input
		size_t input_elements = original.width() * original.height();//number of input elements
		size_t input_size;
		
		if (bit_16) {
			input_size = input_elements * sizeof(unsigned short);
		}
		else {
			input_size = input_elements * sizeof(unsigned char);
		}

		vector<int> B(nr_bins, 0);

		size_t padding_size = input_elements % 32;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> B_ext(32 - padding_size, -1);
			//append that extra vector to our input
			B.insert(B.end(), B_ext.begin(), B_ext.end());
		}

		padding_size = nr_bins % 32;

		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> B_ext(32 - padding_size, -1);
			//append that extra vector to our input
			B.insert(B.end(), B_ext.begin(), B_ext.end());
		}


		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		
		int group_size = B.size();
		size_t output_size = B.size() * sizeof(tvector);//size in bytes

		int max_wg = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

		if (group_size > max_wg && !bit_16) {
			group_size = max_wg;
		}

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_TEMP(context, CL_MEM_WRITE_ONLY, output_size);

		//Part 4 - device operations

		//4.1 copy array A to and initialise other arrays on device memory
		if (bit_16) {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &image_input_16.data()[0]);
		}
		else {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &image_input.data()[0]);
		}
		
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, output_size, &B.data()[0]);//zero B buffer on device memory
		queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, output_size, &B.data()[0]);
		queue.enqueueWriteBuffer(buffer_D, CL_TRUE, 0, output_size, &B.data()[0]);

		//4.2 Setup and execute all kernels (i.e. device code)

		cl::Kernel kernel_1;
		if (!bit_16) {
			kernel_1 = cl::Kernel(program, "histogram");
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(nr_bins * sizeof(int)));//local memory size
			kernel_1.setArg(3, sizeof(cl_int), &nr_bins);
		}
		else {
			kernel_1 = cl::Kernel(program, "histogram_16");
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, sizeof(cl_int), &nr_bins);
		}
		

		cl::Kernel kernel_2;
		if (!bit_16) {
			kernel_2 = cl::Kernel(program, "scan_add");
			kernel_2.setArg(0, buffer_B);
			kernel_2.setArg(1, buffer_C);
			kernel_2.setArg(2, cl::Local(nr_bins * sizeof(int)));
			kernel_2.setArg(3, cl::Local(nr_bins * sizeof(int)));
			kernel_2.setArg(4, sizeof(cl_int), &nr_bins);
		}
		else {
			kernel_2 = cl::Kernel(program, "scan_add_16");
			kernel_2.setArg(0, buffer_B);
			kernel_2.setArg(1, buffer_C);
		}
			

		cl::Kernel kernel_3 = cl::Kernel(program, "normalise");
		kernel_3.setArg(0, buffer_C);
		kernel_3.setArg(1, buffer_D);
		kernel_3.setArg(2, sizeof(cl_int), &input_elements);
		kernel_3.setArg(3, sizeof(cl_int), &nr_bins);

		cl::Kernel kernel_4;
		
		if (!bit_16) {
			kernel_4 = cl::Kernel(program, "apply_lut");
		}
		else {
			kernel_4 = cl::Kernel(program, "apply_lut_16");
		}
		
		kernel_4.setArg(0, buffer_A);
		kernel_4.setArg(1, buffer_D);
		kernel_4.setArg(2, buffer_E);
		kernel_4.setArg(3, sizeof(cl_int), &nr_bins);

		cl::Kernel global_hist = cl::Kernel(program, "histogram_atomic");
		global_hist.setArg(0, buffer_A);
		global_hist.setArg(1, buffer_TEMP);
		global_hist.setArg(2, sizeof(cl_int), &nr_bins);

		cl::Kernel scan_add_atomic = cl::Kernel(program, "scan_add_atomic");
		scan_add_atomic.setArg(0, buffer_B);
		scan_add_atomic.setArg(1, buffer_TEMP);

		cl::Kernel belloch = cl::Kernel(program, "blelloch_scan");
		belloch.setArg(0, buffer_B);
		belloch.setArg(1, buffer_TEMP);
		belloch.setArg(2, cl::Local(nr_bins * sizeof(int)));
		belloch.setArg(3, sizeof(cl_int), &nr_bins);

		cl::Kernel belloch2 = cl::Kernel(program, "scan_bl");
		belloch2.setArg(0, buffer_B);

		cl::Kernel belloch3 = cl::Kernel(program, "scan_bl_local");
		belloch3.setArg(0, buffer_TEMP);
		//belloch3.setArg(1, cl::Local(nr_bins * sizeof(int)));

		cerr << kernel_1.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << std::endl;
		cerr << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

		cl::Event histogram;
		cl::Event cumulative;
		cl::Event normalise;
		cl::Event scan_atomic;
		cl::Event global_hist_prof;
		cl::Event belloch_prof;
		
		//call all kernels in a sequence

		if (bit_16) {
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NullRange, NULL, &histogram);
			queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(group_size), cl::NullRange, NULL, &cumulative);
			queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(group_size), cl::NullRange, NULL, &normalise);
			queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NullRange);
		}
		else {
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(group_size), NULL, &histogram);
			queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &cumulative);
			queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &normalise);
			queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(group_size));
		}
		
		normalise.wait();

		std::cout << "Histogram: "
			<< GetFullProfilingInfo(histogram, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Cumulative: "
			<< GetFullProfilingInfo(cumulative, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Normalise: "
			<< GetFullProfilingInfo(normalise, ProfilingResolution::PROF_US) << std::endl;

		if (!bit_16) {
			queue.enqueueNDRangeKernel(global_hist, cl::NullRange, cl::NDRange(input_elements), cl::NullRange, NULL, &global_hist_prof);
			queue.enqueueNDRangeKernel(scan_add_atomic, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &scan_atomic);
			queue.enqueueNDRangeKernel(belloch2, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &belloch_prof);

			belloch_prof.wait();
			
			std::cout << endl << "---Other methods---" << endl;
			std::cout << "Global Memory Histogram: "
				<< GetFullProfilingInfo(global_hist_prof, ProfilingResolution::PROF_US) << std::endl;
			std::cout << "Scan Add Atomic: "
				<< GetFullProfilingInfo(scan_atomic, ProfilingResolution::PROF_US) << std::endl;
			std::cout << "Blelloch Scan: "
				<< GetFullProfilingInfo(belloch_prof, ProfilingResolution::PROF_US) << std::endl;
		}
		
		//4.3 Copy the result from device to host
		vector<int> hist(group_size);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, group_size*sizeof(int), &hist[0]);

		std::cout << "D = " << hist << std::endl;

		vector<unsigned short> out_16(input_elements, 0);
		vector<unsigned char> out(input_elements, 0);

		if (bit_16) {
			queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, input_size, &out_16.data()[0]);
		}
		else {
			queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, input_size, &out.data()[0]);
		}
		
		CImg<unsigned char> output_image(original.width(), original.height(), original.depth(), original.spectrum());
		output_image.get_shared_channel(0) = CImg<unsigned char>(out.data(), original.width(), original.height(), original.depth(), 1);
		CImg<unsigned short> output_image_16(original.width(), original.height(), original.depth(), original.spectrum());
		output_image_16.get_shared_channel(0) = CImg<unsigned short>(out_16.data(), original.width(), original.height(), original.depth(), 1);

		if (original.spectrum() == 3) {
			output_image.get_shared_channel(1) = cb;
			output_image.get_shared_channel(2) = cr;
			
			output_image = output_image.get_YCbCrtoRGB();
		}

		CImgDisplay disp_output;

		if (bit_16) {
			disp_output = CImgDisplay(output_image_16, "output");
		}
		else {
			disp_output = CImgDisplay(output_image, "output");
		}


		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}


	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}

