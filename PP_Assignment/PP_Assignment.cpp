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
	std::cerr << "  -b : custom bin size" << std::endl;
	std::cerr << "  -m : scan method (0, Hills, 1, Blelloch)" << std::endl;
	std::cerr << "  -o : output intermediate vectors" << std::endl;
}

int main(int argc, char** argv) {
	typedef unsigned char mytype;

	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	int nr_bins = 0;
	int scan_method = 0;
	int output = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if ((strcmp(argv[i], "-m") == 0) && (i < (argc - 1))) { scan_method = atoi(argv[++i]); } // Added arg for scan method
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { nr_bins = atoi(argv[++i]); } // Added arg for custom bin sizes
		else if ((strcmp(argv[i], "-o") == 0) && (i < (argc - 1))) { output = atoi(argv[++i]); } // Added arg for custom bin sizes
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImgDisplay disp_input;

		// Open and read the image file header to find the bit-depth
		// of the image
		bool bit_16 = false;

		cout << image_filename << ", ";

		string f;
		fstream file;
		file.open(image_filename);
		for (int i = 0; i < 100; i++)
		{
			getline(file, f);

			// 8-bit image
			if (f == to_string(255)) {
				disp_input = CImgDisplay(CImg<unsigned char>(image_filename.c_str()), "input");

				cout << "8-bit" << ", ";

				// If not custom bin size or is out of range set to 8-bit max
				if (nr_bins <= 0 || nr_bins > 256) {
					nr_bins = 256;
				}

				break;
			}

			// 16-bit image
			else if (f == to_string(65535)) {
				disp_input = CImgDisplay(CImg<unsigned short>(image_filename.c_str()), "input");
				bit_16 = true;

				cout << "16-bit" << ", ";

				// If not custom bin size or is out of range set to 16-bit max
				if (nr_bins <= 0 || nr_bins > 65536) {
					nr_bins = 65536;
				}

				break;
			}
		}
		
		// Load input image
		CImg<unsigned char> image_input(image_filename.c_str());
		CImg<unsigned short> image_input_16(image_filename.c_str());
		CImg<unsigned short> original;

		if (bit_16) {
			original = image_input_16;
		}
		else {
			original = image_input;
		}
		

		// Check to see if the image is a colour image, if so
		// convert to YCbCr and set the luminance channel as the
		// image input, and copy the other channel so they can be
		// recombined later
		CImg<unsigned char> cb;
		CImg<unsigned char> cr;

		if (image_input.spectrum() == 3) {
			cb = original.get_RGBtoYCbCr().channel(1);
			cr = original.get_RGBtoYCbCr().channel(2);
			image_input = image_input.get_RGBtoYCbCr();
			image_input = image_input.channel(0);

			cout << "Colour, ";
		}
		else {
			cout << "Grayscale, ";
		}

		cout << nr_bins << " bins" << endl;

		if (!bit_16) {
			if (scan_method < 1) {
				cout << "Using Hillis-Steele scan method" << endl;
			}
			else {
				cout << "Using Blelloch scan method" << endl;
			}
		}
		else {
			cout << "Using atomic scan method" << endl;
		}

		// Part 3 - host operations
		// 3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl << endl;

		// Create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// 3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		// Build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Part 3 - memory allocation
		// host - input
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

		// if the input vector is not a multiple of the local_size
		// insert additional neutral elements (0 for addition) so that the total will not be affected
		/*if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> B_ext(32 - padding_size, -1);
			//append that extra vector to our input
			B.insert(B.end(), B_ext.begin(), B_ext.end());
		}*/

		//If bin size is not a multiple of 32 (preferred work group size)
		// then pad the bin vector with -1 values
		padding_size = nr_bins % 32;

		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> B_ext(32 - padding_size, -1);
			//append that extra vector to our input
			B.insert(B.end(), B_ext.begin(), B_ext.end());
		}

		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		
		int group_size = B.size();
		size_t output_size = B.size() * sizeof(int);//size in bytes

		int max_wg = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

		// If the number of bins/work group size is over the maximum size
		// of a work group on the device then 
		//if (group_size > max_wg && !bit_16) {
		//	group_size = max_wg;
		//}

		// Device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_TEMP(context, CL_MEM_WRITE_ONLY, output_size);

		//Part 4 - device operations
		//4.1 copy array A to and initialise other arrays on device memory

		cl::Event im_write_prof;
		cl::Event hist_write_prof;
		//cl::Event 

		if (bit_16) {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &image_input_16.data()[0], NULL, &im_write_prof);
		}
		else {
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &image_input.data()[0], NULL, &im_write_prof);
		}
		
		//queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, output_size, &B.data()[0]);//zero B buffer on device memory
		//queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, output_size, &B.data()[0]);
		//queue.enqueueWriteBuffer(buffer_D, CL_TRUE, 0, output_size, &B.data()[0]);

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
			kernel_2 = cl::Kernel(program, "scan_add_atomic");
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
		belloch3.setArg(0, buffer_B);
		belloch3.setArg(1, buffer_C);
		//belloch3.setArg(1, cl::Local(nr_bins * sizeof(int)));

		//cerr << kernel_1.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << std::endl;
		//cerr << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

		// Declare events for profiling
		cl::Event histogram;
		cl::Event cumulative;
		cl::Event normalise;
		cl::Event map;

		cl::Event scan_atomic;
		cl::Event global_hist_prof;
		cl::Event belloch_prof;
		
		// Call all kernels in a sequence
		if (bit_16) {
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NullRange, NULL, &histogram);
			queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(group_size), cl::NullRange, NULL, &cumulative);
			queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(group_size), cl::NullRange, NULL, &normalise);
			queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NullRange, NULL, &map);
		}
		else {
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(group_size), NULL, &histogram);

			if (scan_method < 1) {
				queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &cumulative);
			}
			else {
				queue.enqueueNDRangeKernel(belloch3, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &cumulative);
			}
			
			queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &normalise);
			queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(group_size), NULL, &map);
		}
		
		map.wait(); // Wait for final kernel to finish
		
		cl::Event im_read_prof;

		// Initialise output vectors
		vector<unsigned short> out_16(input_elements, 0);
		vector<unsigned char> out(input_elements, 0);

		// Copy output data from the device to the host and into the appropriate vector
		if (bit_16) {
			queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, input_size, &out_16.data()[0], NULL, &im_read_prof);
		}
		else {
			queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, input_size, &out.data()[0], NULL, &im_read_prof);
		}

		// Initialise image outputs and set first channel to output data
		CImg<unsigned char> output_image(original.width(), original.height(), original.depth(), original.spectrum());
		output_image.get_shared_channel(0) = CImg<unsigned char>(out.data(), original.width(), original.height(), original.depth(), 1);

		CImg<unsigned short> output_image_16(original.width(), original.height(), original.depth(), original.spectrum());
		output_image_16.get_shared_channel(0) = CImg<unsigned short>(out_16.data(), original.width(), original.height(), original.depth(), 1);


		// If the input image is a colour image, add the colour
		// channels back to the output image and convert it
		// to an RGB image.
		if (original.spectrum() == 3) {
			output_image.get_shared_channel(1) = cb;
			output_image.get_shared_channel(2) = cr;
			
			output_image = output_image.get_YCbCrtoRGB();
		}

		CImgDisplay disp_output; // Initialise output display

		// Display final output image
		if (bit_16) {
			disp_output = CImgDisplay(output_image_16, "output");
		}
		else {
			disp_output = CImgDisplay(output_image, "output");
		}

		// Print profiling results
		std::cout << "Histogram: "
			<< GetFullProfilingInfo(histogram, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Cumulative: "
			<< GetFullProfilingInfo(cumulative, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Normalise: "
			<< GetFullProfilingInfo(normalise, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Map LUT: "
			<< GetFullProfilingInfo(map, ProfilingResolution::PROF_US) << std::endl << endl;

		std::cout << "Image Input vector write time [ns]: " <<
			im_write_prof.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			im_write_prof.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Image Input vector read time [ns]: " <<
			im_read_prof.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			im_read_prof.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl << endl;

		//4.3 Copy the result from device to host
		if (output) {

			// Initialise output vectors
			vector<int> hist(group_size);
			vector<int> cum(group_size);
			vector<int> norm(group_size);

			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, group_size * sizeof(int), &hist[0]);
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, group_size * sizeof(int), &cum[0]);
			queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, group_size * sizeof(int), &norm[0]);

			cout << "Histogram = " << hist << endl << endl;
			cout << "Cumulative = " << cum << endl << endl;
			cout << "LUT = " << norm << endl << endl;
		}

		// If image is 8-bit then run and profile the data against un-optimised
		// and different algorithms/methods 
		if (!bit_16) {
			queue.enqueueNDRangeKernel(global_hist, cl::NullRange, cl::NDRange(input_elements), cl::NullRange, NULL, &global_hist_prof);
			queue.enqueueNDRangeKernel(scan_add_atomic, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &scan_atomic);

			if (scan_method < 1) {
				queue.enqueueNDRangeKernel(belloch3, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &belloch_prof);
			}
			else {
				queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(group_size), cl::NDRange(group_size), NULL, &belloch_prof);
			}

			belloch_prof.wait(); // Wait for final kernel to finish

			// Print profiling results
			std::cout << endl << "---Other methods---" << endl;
			std::cout << "Atomic Histogram: "
				<< GetFullProfilingInfo(global_hist_prof, ProfilingResolution::PROF_US) << std::endl;
			std::cout << "Atomic Scan: "
				<< GetFullProfilingInfo(scan_atomic, ProfilingResolution::PROF_US) << std::endl;

			if (scan_method < 1) {
				std::cout << "Blelloch Scan: ";
			}
			else {
				cout << "Hillis-Steele: ";
			}

			cout << GetFullProfilingInfo(belloch_prof, ProfilingResolution::PROF_US) << std::endl << endl;

		}

		// Close program on ESCAPE key 
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


