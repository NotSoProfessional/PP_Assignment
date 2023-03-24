kernel void normaliseo(global const int* H, global int* N_H, const int im_size, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	local int* scratch_N_H;

	scratch[lid] = H[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	float total = im_size;
	float div = scratch[lid] / total;

	scratch_N_H = scratch;
	scratch_N_H[lid] = div*256;

	N_H[id] = scratch_N_H[lid];
}

kernel void histogram(global const uchar* A, global int* H, local int* L_H, const int nr_bins) {
	int id = get_global_id(0);
	int lid = get_local_id(0);

	int bin_index = A[id] * nr_bins / 256;//take value as a bin index
	//printf("%d\n", A[id]);
	L_H[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&L_H[bin_index]);

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_add(&H[lid], L_H[lid]);//serial operation, not very efficient!
}

kernel void histogram_16(global const ushort* A, global int* H, const int nr_bins) {
	int id = get_global_id(0);

	int bin_index = A[id] * nr_bins / 65535;//take value as a bin index
	//printf("%d, ", bin_index);
	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

kernel void histogram_atomic(global const uchar* A, global int* H, const int nr_bins) {
	int id = get_global_id(0);

	int bin_index = A[id] * nr_bins / 256;//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

kernel void scan_add(global const int* A, global int* B, local int* scratch_1, local int* scratch_2, const int bins) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = bins;
	local int* scratch_3;//used for buffer swap


	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	//printf("%d\n", lid);
	for (int i = 1; i < N; i *= 2) {
		if (lid >= i) {
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		}

		else {
			scratch_2[lid] = scratch_1[lid];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;

	}


	//copy the cache to output array
	if (lid < bins) {
		B[id] = scratch_1[lid];
	}
}

kernel void scan_add_16(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = id + 1; i < N && id < N; i++) {
		if (B[i] != -1) {
			atomic_add(&B[i], A[id]);
		}
	}
}

kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N && id < N; i++)
		atomic_add(&B[i], A[id]);
}

kernel void normalise(global const int* H, global int* N_H, const int im_size, const int nr_bins) {
	int id = get_global_id(0);
	float total = im_size;

	if (H[id] != -1) {
		N_H[id] = (H[id] / total) * (nr_bins - 1);
	}
}


kernel void normalise_16(global const int* H, global int* N_H, const int im_size, const int nr_bins) {
	int id = get_global_id(0);
	float total = im_size;

	if (H[id] != -1) {
		N_H[id] = (H[id] / total) * (nr_bins - 1);
	}
}

kernel void apply_lut(global const uchar* I, global const int* LUT, global uchar* O, const int nr_bins) {
	int id = get_global_id(0);
	float bins = 255;
	float t_bins = nr_bins;
	int index = I[id] * (t_bins / bins);

	//int val_new = LUT[index] * (bins/nr_bins);
	uchar val_new = LUT[index] * (bins / (nr_bins-1));
	
	O[id] = val_new;
}

kernel void apply_lut_16(global const ushort* I, global const int* LUT, global ushort* O, const int nr_bins) {
	int id = get_global_id(0);
	float bins = 65535;
	float t_bins = nr_bins;
	int index = I[id] * (t_bins / bins);

	ushort val_new = LUT[index] * (bins / (nr_bins - 1));

	O[id] = val_new;
}

kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	// Up-sweep phase
	for (int stride = 1; stride < N; stride *= 2) {
		int idx = (id + 1) * stride * 2 - 1;
		if (idx < N) {
			A[idx] += A[idx - stride];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// Down-sweep phase
	if (id == 0) {
		A[N - 1] = 0;
	}

	//printf("%d, %d\n", id, A[id]);

	barrier(CLK_GLOBAL_MEM_FENCE);

	// Downsweep
	for (int stride = N / 2; stride > 0; stride /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		int idx = 2 * (id + 1) * stride - 1;
		if (idx < N) {
			int temp = A[idx - stride];
			A[idx - stride] = A[idx];
			A[idx] += temp;
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);


	//printf("%d, %d\n", id, A[id]);
}

kernel void scan_bl_local(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	local int lA[2048];

	lA[id] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// Up-sweep phase
	for (int stride = 1; stride < N; stride *= 2) {
		int idx = (id + 1) * stride * 2 - 1;
		if (idx < N) {
			lA[idx] += lA[idx - stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Down-sweep phase
	if (id == 0) {
		lA[N - 1] = 0;
	}

	//printf("%d, %d\n", id, A[id]);

	barrier(CLK_LOCAL_MEM_FENCE);

	// Downsweep
	for (int stride = N / 2; stride > 0; stride /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		int idx = 2 * (id + 1) * stride - 1;
		if (idx < N) {
			int temp = lA[idx - stride];
			lA[idx - stride] = lA[idx];
			lA[idx] += temp;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	A[id] = lA[id];
	//printf("%d, %d\n", id, A[id]);
}

kernel void blelloch_scan(global const int* input, global int* output, local int* local_data, const int n)
{
	int tid = get_local_id(0);
	int gid = get_global_id(0);

	// Load input data into local memory
	local_data[tid] = (gid < n) ? input[gid] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Reduction phase
	for (int stride = 1; stride <= get_local_size(0); stride *= 2)
	{
		int index = (tid + 1) * stride * 2 - 1;
		if (index < get_local_size(0) && index >= stride)
		{
			int val1 = local_data[index];
			int val2 = local_data[index - stride];
			if (val1 != -1 && val2 != -1)
			{
				local_data[index] += val2;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Post reduction phase
	for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		int index = (tid + 1) * stride * 2 - 1;
		if (index + stride < get_local_size(0))
		{
			int val1 = local_data[index];
			int val2 = local_data[index + stride];
			if (val1 != -1 && val2 != -1)
			{
				local_data[index + stride] += val1;
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Write output data to global memory
	if (gid < n && local_data[tid] != -1)
	{
		output[gid] = local_data[tid];
	}
}

kernel void blelloch_scano(global int* input, global int* output, local int* local_data, const int n)
{
	int tid = get_local_id(0);
	int gid = get_global_id(0);

	// Load input data into local memory
	local_data[tid] = (gid < n) ? input[gid] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Reduction phase
	for (int stride = 1; stride <= get_local_size(0); stride *= 2)
	{
		int index = (tid + 1) * stride * 2 - 1;
		if (index < get_local_size(0))
		{
			local_data[index] += local_data[index - stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Post reduction phase
	for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		int index = (tid + 1) * stride * 2 - 1;
		if (index + stride < get_local_size(0))
		{
			local_data[index + stride] += local_data[index];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Write output data to global memory
	if (gid < n)
	{
		output[gid] = local_data[tid];
	}
}
