
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <utility>
#include <fstream>
#include <stdio.h>
#include <string>
#include <chrono>

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif


#define dd double
#define threads 32


const dd G = 6.674e-11;
const dd dt = 0.001;

class MatPoint {
public:
    dd x;
    dd y;
    dd vx;
    dd vy;
    dd m;
};

class DDPair {
public:
    dd x;
    dd y;
};


cudaError_t oneMemomoryMethod(int* c, const int* a, const int* b, unsigned int size);
char* twoMutexMethod(MatPoint* points, unsigned period, unsigned int size, char* log_buffer_, DDPair* ddPairs);


void copyMatPoint(MatPoint* source, MatPoint* dist) {
    dist->x = source->x;
    dist->y = source->y;
    dist->vx = source->vx;
    dist->vy = source->vy;
    dist->m = source->m;
}

MatPoint* vectorToArray(std::vector<MatPoint> vector) {
    MatPoint* arrayPoints = new MatPoint[vector.size()];
    for (int i = 0; i < vector.size(); ++i) {
        copyMatPoint(&vector[i], &arrayPoints[i]);
    }
    return arrayPoints;
}

__global__ void pointRoutinue(MatPoint* points, char* log_buffer, unsigned points_num, unsigned threads_num, unsigned steps, DDPair* result_calc, DDPair* calc) {
    const dd G = 6.674e-11;
    const dd dt = 0.001;

    unsigned index = index = blockIdx.x * blockDim.x + threadIdx.x;

    int number_of_point = points_num / threads;
    unsigned start_index = index * number_of_point;
    unsigned end_index = (index + 1) * number_of_point;

    if (index == (threads - 1))
        end_index = points_num;

    for (unsigned i = 0; i < steps; ++i) {
        for (unsigned j = start_index, result_index = 0; j < end_index; ++j, ++result_index) {
            dd sum_x = 0;
            dd sum_y = 0;
            for (unsigned k = 0; k < points_num; ++k) {
                if (j == k) {
                    continue;
                }
                dd dist = sqrt(pow((points[k].x - points[j].x), 2) + pow((points[k].y - points[j].y), 2));
                sum_x += points[k].m * (points[k].x - points[j].x) / pow(dist, 3);
                sum_y += points[k].m * (points[k].y - points[j].y) / pow(dist, 3);
            }
            calc[j].x = G * points[j].m * sum_x;
            calc[j].y = G * points[j].m * sum_y;
        }

        cuda_SYNCTHREADS();

        for (unsigned j = start_index, result_index = 0; j< end_index; ++j, ++result_index) {
            points[j].vx += calc[j].x / points[j].m * dt;
            points[j].vy += calc[j].y / points[j].m * dt;
            points[j].x += points[j].vx * dt;
            points[j].y += points[j].vy * dt;
        }

        for (unsigned j = start_index; j < end_index; ++j) {

            result_calc[(i * (points_num)+j)].x = points[j].x;
            result_calc[(i * (points_num)+j)].y = points[j].y;
            //char log[40];
            //memccpy(log, (void*)points[j + 1]->x, 40 * sizeof(char))
            //memcpy(log, &(points[j].x), sizeof(points[j].x));
            //log[sizeof(points[j].x)] = ',';
            //memcpy(&log[sizeof(points[j].x)+1], &(points[j].y), sizeof(points[j].y));
            //log[sizeof(points[j].x) + sizeof(points[j].y) + 1] = ',';
            //memcpy(&log_buffer[(i * (points_num + 2)) * 40 + j * 40 * sizeof(char)], log, 40 * sizeof(char));
        }
        cuda_SYNCTHREADS();
    }
}


void read_file(std::vector<MatPoint>& points) {
    std::ifstream file("input.txt");
    dd x, y, vx, vy, m;
    while (!file.eof()) {
        file >> x >> y >> vx >> vy >> m;
        points.push_back({ x, y, vx, vy, m });
    }
}

void write_results(std::ofstream& file, char* log_buffer, unsigned period, unsigned points_num) {
    char* log = new char[40];
    double* log_double;
    int j = 0;
    int points = 0;
    int line_points = 0;
    for (long long i = 0; i < (unsigned)(period / dt) * (points_num + 2) * 40 * sizeof(char); ++i) {
        if (line_points == 17) {
            points = 0;
            line_points = 0;
            j = 0;
            file << '\n';
        }
        if (log_buffer[i] == ',') {
            if (log_buffer[i] == ',' && j != 0) {
                std::string log_str;
                memcpy(log, &log_buffer[i - j], (j + 1) * sizeof(char));
                log[j] = '\0';
                log_str = std::to_string(*(double*)log);
                file << log_str + ',';
            }
            points++;
            line_points++;
            j = 0;
            continue;
        }

        if ((points == 2 && i % 40 != 0)) {
            continue;
        }
        else if (points == 2 && i % 40 == 0)
        {
            points = 0;
        }
        ++j;
    }
}

void write_resultsSTR(std::ofstream& file, DDPair* log_buffer, unsigned period, unsigned points_num) {
    for (unsigned i = 0; i < (unsigned)period / dt; ++i) {
        file << (double)i / dt << ",";
        for (unsigned j = 0; j < points_num; ++j) {
            file << log_buffer[i * points_num + j].x << "," << log_buffer[i * points_num + j].y << ", ";
        }
        file << "\n";
    }
}


int main() {
    std::vector<MatPoint> points;
    read_file(points);
    cudaError_t cudaStatus;
    char* log_buffer = 0;
    unsigned period = 1;

    std::ofstream file("output.txt");
    std::ofstream file_2("outputSTR.txt");
    file << "t,";
    for (unsigned i = 0; i < points.size(); ++i) {
        file << "x" << i + 1 << ",y" << i + 1 << ",";
    }
    file << "\n";

    file_2 << "t,";
    for (unsigned i = 0; i < points.size(); ++i) {
        file_2 << "x" << i + 1 << ",y" << i + 1 << ",";
    }
    file_2 << "\n";

    MatPoint* mathPoints = vectorToArray(points);
    unsigned points_num = points.size();
    DDPair* ddPairs = new DDPair[(unsigned)(period / dt) * points_num * sizeof(DDPair)];
    points.clear();
    std::chrono::time_point<std::chrono::steady_clock> begin, end;
    begin = std::chrono::steady_clock::now();
    log_buffer = twoMutexMethod(mathPoints, period, points_num, log_buffer, ddPairs);
    end = std::chrono::steady_clock::now();
    std::cout << "Time with Cuda: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";

    write_results(file, log_buffer, period, points_num);
    write_resultsSTR(file_2, ddPairs, period, points_num);
    

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    if (mathPoints) {
        delete(mathPoints);
    }

    if (log_buffer) {
        delete(log_buffer);
    }

    delete(ddPairs);
  

    return 0;
}



char* twoMutexMethod(MatPoint* points, unsigned period, unsigned int size, char* log_buffer_, DDPair* ddPairs)
{
    MatPoint* dev_points = 0;
    char* log_buffer = 0;
    cudaError_t cudaStatus;
    DDPair* dev_calc_results = 0;
    DDPair* dev_results = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_points, size * sizeof(MatPoint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&log_buffer,  (unsigned)(period / dt) * (size + 2) * 40 * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_points, points, size * sizeof(MatPoint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_calc_results, (unsigned)(period / dt) * size * sizeof(DDPair));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_results, size * sizeof(DDPair));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    log_buffer_ = new char[(unsigned)(period / dt) * (size + 2) * 40 * sizeof(char)];

    // Launch a kernel on the GPU with one thread for each element.
    pointRoutinue << <1, threads >> > (dev_points, log_buffer, size, threads, (period / dt), dev_calc_results, dev_results);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(log_buffer_, log_buffer, (unsigned)(period / dt) * (size + 2) * 40 * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    //for (unsigned i = 0; i < (unsigned)(period / dt); ++i) {
    //    strcpy(&log_buffer_[(i * (size + 2)) * 40 + 0 * 40 * sizeof(char)], std::string(std::to_string(i * dt) + ",,").c_str());
    //    strcpy(&log_buffer_[(i * (size + 2)) * 40 + (size + 1) * 40 * sizeof(char)], "\n");
    //}

    cudaStatus = cudaMemcpy(ddPairs, dev_calc_results, (unsigned)(period / dt) * size * sizeof(DDPair), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    
Error:
    if (dev_points) {
        cudaFree(dev_points);
    }

    if (dev_calc_results) {
        cudaFree(dev_calc_results);
    }

    if (dev_results) {
        cudaFree(dev_results);
    }

    if (log_buffer) {
        cudaFree(log_buffer);
    }

    return log_buffer_;
}
