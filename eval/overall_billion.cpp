/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <thread>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <assert.h>

#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;
double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


bool file_exist(const std::string& file_path)
{
	if (FILE* file = fopen(file_path.c_str(), "r")){
		fclose(file);
		return true;
	}
	else 
		return false;
}

double inter_sec(faiss::Index::idx_t *taget, int *gt, int k){
    double res = 0.;
    for (int i = 0; i < k; i++){
        int val = taget[i];
        for (int j = 0; j < k; j++){
            if (val == gt[j]){
                res += 1.;
                break;
            }
        }
    }
    return res / k;
}

float* bvecs_read_10(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, 1, sizeof(int), f);
    assert(d > 0 && d < 1000000);

    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    assert(sz % (d + 4) == 0);  // each record: 4 bytes for int + d bytes for uint8
    size_t n = sz / (d + 4);
    n = n / 100;
    printf("bvecs_read: d : %d, n: %d\n", d, n);

    *d_out = d;
    *n_out = n;

    // allocate temp buffer to hold raw bvecs
    uint8_t* buf = new uint8_t[n * (d + 4)];
    size_t nr = fread(buf, 1, n * (d + 4), f);
    assert(nr == n * (d + 4));

    fclose(f);

    // allocate float array to hold final result
    float* x = new float[n * d];

    for (size_t i = 0; i < n; i++) {
        uint8_t* vec = buf + i * (d + 4);
        // skip first 4 bytes (dimension) and convert uint8 to float
        for (int j = 0; j < d; j++) {
            x[i * d + j] = (float)vec[4 + j];
        }
    }

    delete[] buf;
    return x;
}

float* bvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, 1, sizeof(int), f);
    assert(d > 0 && d < 1000000);

    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    assert(sz % (d + 4) == 0);  // each record: 4 bytes for int + d bytes for uint8
    size_t n = sz / (d + 4);
    printf("bvecs_read: d : %d, n: %d\n", d, n);

    *d_out = d;
    *n_out = n;

    // allocate temp buffer to hold raw bvecs
    uint8_t* buf = new uint8_t[n * (d + 4)];
    size_t nr = fread(buf, 1, n * (d + 4), f);
    assert(nr == n * (d + 4));

    fclose(f);

    // allocate float array to hold final result
    float* x = new float[n * d];

    for (size_t i = 0; i < n; i++) {
        uint8_t* vec = buf + i * (d + 4);
        // skip first 4 bytes (dimension) and convert uint8 to float
        for (int j = 0; j < d; j++) {
            x[i * d + j] = (float)vec[4 + j];
        }
    }

    delete[] buf;
    return x;
}

std::vector<float*> bvecs_reads(const char* fname, size_t* d_out, size_t* n_out, int slice = 100) {
    std::vector<float*> vec(slice);

    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, sizeof(int), 1, f);
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fclose(f);

    assert(d > 0 && d < 1000000);
    assert(sz % (d + 4) == 0 && "invalid file size for .bvecs");

    size_t n = sz / (d + 4);
    *d_out = d;
    *n_out = n;
    printf("bvecs_reads: d : %d, n: %d\n", d, n);

    int64_t total_size = static_cast<int64_t>(n);
    int64_t slice_size = total_size / slice;

#pragma omp parallel for
    for (int i = 0; i < slice; i++) {
        auto t0 = omp_get_wtime();
        FILE* f = fopen(fname, "r");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fname);
            perror("");
            abort();
        }

        int64_t start = (d + 4) * i * slice_size;
        fseek(f, start, SEEK_SET);

        // 마지막 슬라이스는 남은 만큼 처리
        int64_t this_slice = (i == slice - 1) ? (total_size - slice_size * i) : slice_size;
        uint8_t* buf = new uint8_t[this_slice * (d + 4)];
        size_t nr = fread(buf, 1, this_slice * (d + 4), f);
        fclose(f);

        assert(nr == static_cast<size_t>(this_slice * (d + 4)));

        float* x = new float[this_slice * d];
        for (int64_t j = 0; j < this_slice; j++) {
            uint8_t* vec = buf + j * (d + 4);
            for (int k = 0; k < d; k++) {
                x[j * d + k] = static_cast<float>(vec[4 + k]);
            }
        }

        delete[] buf;
        vec[i] = x;

        auto t1 = omp_get_wtime();
#pragma omp critical
        {
            printf("Read %d/%d done... , Thread %d : %.3f s\n", i, slice, omp_get_thread_num(), t1 - t0);
        }
    }

    return vec;
}

std::vector<float*> bvecs_reads_10(const char* fname, size_t* d_out, size_t* n_out, int slice = 100) {
    std::vector<float*> vec(slice);

    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, sizeof(int), 1, f);
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fclose(f);

    assert(d > 0 && d < 1000000);
    assert(sz % (d + 4) == 0 && "invalid file size for .bvecs");

    size_t n = sz / (d + 4);

    n = n / 100;

    *d_out = d;
    *n_out = n;
    printf("bvecs_reads: d : %d, n: %d\n", d, n);

    int64_t total_size = static_cast<int64_t>(n);
    int64_t slice_size = total_size / slice;

#pragma omp parallel for
    for (int i = 0; i < slice; i++) {
        auto t0 = omp_get_wtime();
        FILE* f = fopen(fname, "r");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fname);
            perror("");
            abort();
        }

        int64_t start = (d + 4) * i * slice_size;
        fseek(f, start, SEEK_SET);

        // 마지막 슬라이스는 남은 만큼 처리
        int64_t this_slice = (i == slice - 1) ? (total_size - slice_size * i) : slice_size;
        uint8_t* buf = new uint8_t[this_slice * (d + 4)];
        size_t nr = fread(buf, 1, this_slice * (d + 4), f);
        fclose(f);

        assert(nr == static_cast<size_t>(this_slice * (d + 4)));

        float* x = new float[this_slice * d];
        for (int64_t j = 0; j < this_slice; j++) {
            uint8_t* vec = buf + j * (d + 4);
            for (int k = 0; k < d; k++) {
                x[j * d + k] = static_cast<float>(vec[4 + k]);
            }
        }

        delete[] buf;
        vec[i] = x;

        auto t1 = omp_get_wtime();
#pragma omp critical
        {
            printf("Read %d/%d done... , Thread %d : %.3f s\n", i, slice, omp_get_thread_num(), t1 - t0);
        }
    }

    return vec;
}
std::vector<float*> bvecs_reads_100(const char* fname, size_t* d_out, size_t* n_out, int slice = 100) {
    std::vector<float*> vec(slice);

    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int d;
    fread(&d, sizeof(int), 1, f);
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fclose(f);

    assert(d > 0 && d < 1000000);
    assert(sz % (d + 4) == 0 && "invalid file size for .bvecs");

    size_t n = sz / (d + 4);

    n = n / 10;

    *d_out = d;
    *n_out = n;
    printf("bvecs_reads: d : %d, n: %d\n", d, n);

    int64_t total_size = static_cast<int64_t>(n);
    int64_t slice_size = total_size / slice;

#pragma omp parallel for
    for (int i = 0; i < slice; i++) {
        auto t0 = omp_get_wtime();
        FILE* f = fopen(fname, "r");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fname);
            perror("");
            abort();
        }

        int64_t start = (d + 4) * i * slice_size;
        fseek(f, start, SEEK_SET);

        // 마지막 슬라이스는 남은 만큼 처리
        int64_t this_slice = (i == slice - 1) ? (total_size - slice_size * i) : slice_size;
        uint8_t* buf = new uint8_t[this_slice * (d + 4)];
        size_t nr = fread(buf, 1, this_slice * (d + 4), f);
        fclose(f);

        assert(nr == static_cast<size_t>(this_slice * (d + 4)));

        float* x = new float[this_slice * d];
        for (int64_t j = 0; j < this_slice; j++) {
            uint8_t* vec = buf + j * (d + 4);
            for (int k = 0; k < d; k++) {
                x[j * d + k] = static_cast<float>(vec[4 + k]);
            }
        }

        delete[] buf;
        vec[i] = x;

        auto t1 = omp_get_wtime();
#pragma omp critical
        {
            printf("Read %d/%d done... , Thread %d : %.3f s\n", i, slice, omp_get_thread_num(), t1 - t0);
        }
    }

    return vec;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

std::vector<float *> fvecs_reads(const char* fname, size_t* d_out, size_t* n_out, int slice = 10){
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    std::vector<float *> res;
    size_t nr = 0;
    size_t slice_size = n / slice * (d + 1);
    size_t total_size = size_t(d + 1) * size_t(n);

    for (int i = 0; i < slice; i++){
        float* x = new float[slice_size];
        nr += fread(x, sizeof(float), slice_size, f);
        for (size_t j = 0; j < n / slice; j++)
            memmove(x + j * d, x + 1 + j * (d + 1), d * sizeof(*x));
        res.push_back(x);
    }

    assert(nr == total_size || !"could not read whole file");
    fclose(f);
    return res;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

int* ibin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        return nullptr;
    }

    uint32_t n = 0, d = 0;
    fread(&n, sizeof(uint32_t), 1, f);
    fread(&d, sizeof(uint32_t), 1, f);

    printf("ibin_read: n = %u, d = %u\n", n, d);
    assert(n > 0 && d > 0 && d < 1000000);

    *n_out = static_cast<size_t>(n);
    *d_out = static_cast<size_t>(d);

    size_t total_size = static_cast<size_t>(n) * static_cast<size_t>(d);
    int* buf = new int[total_size];

    size_t nr = fread(buf, sizeof(int32_t), total_size, f);
    fclose(f);

    assert(nr == total_size && "could not read entire ibin vector_array");
    return buf;
}

float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        return nullptr;
    }

    int d, n;
    fread(&n, sizeof(int), 1, f);
    fread(&d, sizeof(int), 1, f);
    fclose(f);

    printf("fbin_read: d = %d, n = %d\n", d, n);
    assert(d > 0 && d < 1000000);

    *d_out = d;
    *n_out = n;

    size_t total_size = static_cast<size_t>(d) * static_cast<size_t>(n);
    float* x = new float[total_size];

    f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not reopen %s\n", fname);
        perror("");
        delete[] x;
        return nullptr;
    }

    fseek(f, 8, SEEK_SET);  // skip d and n
    size_t nr = fread(x, sizeof(float), total_size, f);
    fclose(f);

    assert(nr == total_size && "could not read whole file");
    return x;
}


std::vector<float*> fbin_reads(const char* fname, size_t* d_out, size_t* n_out, int slice = 100) {
    std::vector<float*> vec(slice);
    FILE* f = fopen(fname, "r");
    int d, n;
    fread(&n, sizeof(int), 1, f);
    fread(&d, sizeof(int), 1, f);
    fclose(f);
    printf("d : %d, n: %d\n", d, n);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    *d_out = d;
    *n_out = n;
    int64_t total_size = int64_t(d) * int64_t(n);
    int64_t slice_size = total_size / slice;
    int num = 0;
#pragma omp parallel for
    for (int i = 0; i < slice; i++){
        auto t0 = elapsed();
        FILE* f = fopen(fname, "r");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fname);
            perror("");
            abort();
        }
        int64_t nr = 0;
        int64_t start = slice_size * i * sizeof(float) + 8;
        fseek(f, start, SEEK_SET);
        float *x = new float[slice_size];
        nr += fread(x, sizeof(float), slice_size, f);
        vec[i] = x;
        auto t1 = elapsed();
        int id = omp_get_thread_num();
        #pragma critical
        {
            printf("Read %d/%d slice done... , Thread %d : %.3f s\n", i, slice, id, t1 - t0);
            printf("Read %d/%d done\n", num++, slice);
        }

        // int64_t nr = fread(x, sizeof(float), total_size, f);
        // printf("Read finished, read %ld\n", nr);
        // assert(nr == total_size || !"could not read whole file");
        fclose(f);
    }
    return vec;
}

// ./script dataset-name bs topk nprobe (./overall deep)
int main(int argc,char **argv){
    std::cout << argc << " arguments" <<std::endl;
    if(argc - 1 != 1){
        printf("You should at least input 3 params: the dataset name, batch size and topk\n");
        return 0;
    }
    std::string p1 = argv[1];

	#pragma omp parallel
    {
        // 병렬 영역에서 실제 스레드 수 출력(한 번만)
        #pragma omp single
        std::cout << "[parallel] num_threads = " << omp_get_num_threads() << "\n";
    }

	int num_cores = std::thread::hardware_concurrency();
	printf("num cores: %d\n", num_cores);

    int ncentroids;

    std::string db, train_db, query, gtI;
    int dim;
	if (p1 == "sift"){
        /*
        db = "/fast-lab-share/mhkim/anns-dataset/sift1b/bigann_base.bvecs";
        train_db = "/fast-lab-share/mhkim/anns-dataset/sift1b/bigann_learn.bvecs";
        query = "/fast-lab-share/mhkim/anns-dataset/sift1b/bigann_query.bvecs";
        gtI = "/fast-lab-share/mhkim/anns-dataset/sift1b/gnd/idx_1000M.ivecs";
        */
        db = "/storage/anns-dataset/sift1b/bigann_base.bvecs";
        train_db = "/storage/anns-dataset/sift1b/bigann_learn.bvecs";
        query = "/storage/anns-dataset/sift1b/bigann_query.bvecs";
        gtI = "/storage/anns-dataset/sift1b/gnd/idx_1000M.ivecs";
        dim = 128;
        //ncentroids = 256;
        ncentroids = 32768;
    }
    else if (p1 == "sift10"){
        db = "/storage/anns-dataset/sift1b/bigann_base.bvecs";
        train_db = "/storage/anns-dataset/sift1b/bigann_base.bvecs";
        query = "/storage/anns-dataset/sift1b/bigann_query.bvecs";
        gtI = "/storage/anns-dataset/sift1b/gnd/idx_10M.ivecs";
        dim = 128;
        //ncentroids = 256;
        ncentroids = 32768;
    }
    else if (p1 == "sift100"){
        db = "/storage/anns-dataset/sift1b/bigann_base.bvecs";
        train_db = "/storage/anns-dataset/sift1b/bigann_learn.bvecs";
        query = "/storage/anns-dataset/sift1b/bigann_query.bvecs";
        gtI = "/storage/anns-dataset/sift1b/gnd/idx_100M.ivecs";
        dim = 128;
        //ncentroids = 256;
        ncentroids = 32768;
    }
    else if (p1 == "deep"){
        /*
        db = "/fast-lab-share/mhkim/anns-dataset/deep1b/base.1B.fbin";
        train_db = "/fast-lab-share/mhkim/anns-dataset/deep1b/learn.350M.fbin";
        query = "/fast-lab-share/mhkim/anns-dataset/deep1b/query.public.10K.fbin";
        gtI = "/fast-lab-share/mhkim/anns-dataset/deep1b/groundtruth.public.10K.ibin";
        */
        db = "/storage/anns-dataset/deep1b/base.1B.fbin";
        train_db = "/storage/anns-dataset/deep1b/learn.350M.fbin";
        query = "/storage/anns-dataset/deep1b/query.public.10K.fbin";
        gtI = "/storage/anns-dataset/deep1b/groundtruth.public.10K.ibin";
        dim = 96;
		//ncentroids = 384;
        ncentroids = 32768;
    }
    else if (p1 == "text"){
        /*
        db = "/fast-lab-share/mhkim/anns-dataset/text1b/base.1B.fbin";
        train_db = "/fast-lab-share/mhkim/anns-dataset/text1b/query.learn.50M.fbin";
        query = "/fast-lab-share/mhkim/anns-dataset/text1b/query.public.100K.fbin";
        gtI = "/fast-lab-share/mhkim/anns-dataset/text1b/groundtruth.public.100K.ibin";
        */
        db = "/storage/anns-dataset/text1b/base.1B.fbin";
        train_db = "/storage/anns-dataset/text1b/query.learn.50M.fbin";
        query = "/storage/anns-dataset/text1b/query.public.100K.fbin";
        gtI = "/storage/anns-dataset/text1b/groundtruth.public.100K.ibin";
        dim = 200;
        //ncentroids = 192;
        ncentroids = 32768;
    }
    else{
        printf("Your input dataset is not included yet! \n");
        return 0;
    }

    auto t0 = elapsed();

    omp_set_num_threads(num_cores);

    int dev_no = 0;
    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = dev_no;
    faiss::gpu::GpuIndexIVFFlat *index;

    if (p1 == "text" || p1 == "text30"){
        index = new faiss::gpu::GpuIndexIVFFlat(
            &resources, dim, ncentroids, faiss::METRIC_INNER_PRODUCT, config);
    }
    else{
        index = new faiss::gpu::GpuIndexIVFFlat(
            &resources, dim, ncentroids, faiss::METRIC_L2, config);
    }

    size_t d;
    // Train the index
    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
		float* xt;
		if (p1 == "sift" || p1 == "sift100") {
			xt = bvecs_read(train_db.c_str(), &d, &nt);
        } else if (p1 == "sift10") {
			xt = bvecs_read_10(train_db.c_str(), &d, &nt);
		} else {
			xt = fbin_read(train_db.c_str(), &d, &nt);
		}

        FAISS_ASSERT(d == dim);
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->train(nt, xt);
        delete[] xt;
    }

    // Add the data
    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        int slice = 100;
        omp_set_num_threads(10);
		std::vector<float *> xbs;
		if (p1 == "sift") {
			xbs = bvecs_reads(db.c_str(), &d2, &nb, slice);
        } else if (p1 == "sift100") {
			xbs = bvecs_reads_100(db.c_str(), &d2, &nb, slice);
        } else if (p1 == "sift10") {
			xbs = bvecs_reads_10(db.c_str(), &d2, &nb, slice);
        } else
			xbs = fbin_reads(db.c_str(), &d2, &nb, slice);
        omp_set_num_threads(num_cores);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               nb,
               d);

        for (int i = 0; i < slice; i++){
            double tt0 = elapsed();
            index->add(nb / slice, xbs[i]);
            delete[] xbs[i];
            double tt1 = elapsed();
            printf("Index %d/%d done : %.3f s\n", i, slice, tt1 - tt0);
        }
    }

    size_t nq;
    float* xq;
    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
		if (p1 == "sift" || p1 == "sift100" || p1 == "sift10")
        	xq = bvecs_read(query.c_str(), &d2, &nq);
		else
        	xq = fbin_read(query.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;                // nb of results per query in the GT
    int* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
		int* gt_int; 
		if (p1 == "sift" || p1 == "sift100" || p1 == "sift10")
			gt_int = ivecs_read(gtI.c_str(), &k, &nq2);
		else
			gt_int = ibin_read(gtI.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new int[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

	nq = 8192;
	std::vector<int> topks = {10, 32};
	//std::vector<int> nprobes = {2, 4, 6, 8, 10, 12, 14, 16};
	//std::vector<int> nprobes = {8, 16, 24, 32, 40, 48, 56, 64};
	std::vector<int> batch_sizes = {8192, 1024, 2048, 4096, 8192};

    bf16* xq_bf16 = new bf16[dim * nq];
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < dim; j++)
            xq_bf16[dim * i + j] = __float2bfloat16_rn(xq[dim * i + j]);
    }

	for (int input_k : topks) {
        int in_probe;
        if (p1 == "text"){
            if (input_k == 10)
                in_probe = 64;
            else
                in_probe = 128;
        } else {
            if (input_k == 10)
                in_probe = 32;
            else
                in_probe = 48;
        }
		for (int bs : batch_sizes) {
    		std::vector<float> dis(nq * input_k);
    		std::vector<faiss::Index::idx_t> idx(nq * input_k);
    		index->nprobe = in_probe;
    		double tt0, tt1, total = 0.;

    		int i;
    		for (i = 0; i < nq / bs; i++){
        		tt0 = elapsed();
        		index->search(bs, xq + d * (bs * i), input_k, dis.data() + input_k * (bs * i), idx.data() + input_k * (bs * i));
        		//index->search(bs, xq_bf16 + d * (bs * i), input_k, dis.data() + input_k * (bs * i), idx.data() + input_k * (bs * i));
        		tt1 = elapsed();
            	total += (tt1 - tt0) * 1000;
    		}

    		double acc = 0.;
    		for (int j = 0; j < i * bs; j++){
        		acc += inter_sec(idx.data() + input_k * j, gt + k * j, input_k);
    		}
            acc = acc * 100 / (i*bs);


                printf("=======================================\n");
			    printf("topk: %d, batch size: %d, nprobe: %d\n", input_k, bs, in_probe);
    		    printf("QPS: %.3f\n", nq / (total / 1000));
    		    printf("Ave accuracy : %.1f%% \n", acc);
			    printf("=======================================\n");
		}
	}

    delete[] xq;
	delete[] xq_bf16;
    delete[] gt;
    delete index;

    return 0;
}
