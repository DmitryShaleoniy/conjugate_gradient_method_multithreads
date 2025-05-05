#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <atomic>
#define demension 10000

//функция вычисления скалярного произведения 
double skalar (double *var1, double* var2) {//8 потоков
  omp_set_num_threads(8);
  double res = 0;
  double th_tmp[8] {0};
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for shared(var1, var2)
  for(int i = 0; i < demension; i++) {
    if(i == 0 || i == demension/2 || i == demension*3/4) {
      printf("thread num = %d\n", omp_get_thread_num());
    }
    th_tmp[omp_get_thread_num()] += (*(var1 + i))*(*(var2 + i));
  }
#pragma omp parallel for// reduction(+:res)
for(int i = 0; i < 8; i++){
  res += th_tmp[i];
}
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "time duration multi = " << duration.count() << std::endl;
  return res;
}//работа ядер сводится к ожиданию

double skalar_try (double *var1, double* var2) {//8 потоков, использование reduction явно
  omp_set_num_threads(8);
  double res = 0;
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for shared(var1, var2) reduction(+:res)
  for(int i = 0; i < demension; i++) {
    if(i == 0 || i == demension/2 || i == demension*3/4) {
      printf("thread num = %d\n", omp_get_thread_num());
    }
    res += (*(var1 + i))*(*(var2 + i));
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "time duration multi = " << duration.count() << std::endl;
  return res;
}//работа ядер сводится к ожиданию

double skalar_mono (double *var1, double* var2) { //в одном потоке
  double res = 0;
  for(int i = 0; i < demension; i++) {
    if(i == 0 || i == demension/2 || i == demension*3/4) {
      printf("thread num = %d\n", omp_get_thread_num());
    }
    res += (var1[i] * var2[i]);
  }
  return res;
}

void skalar_test (){
  double var1[demension];
  double var2[demension];
  double res_mono;
  double res_mult;
  time_t start_time, end_time, start_time_mono, end_time_mono;
  #pragma omp parallel for
  for (int i = 0; i < demension; i++){
    *(var1 + i) = i;
    *(var2 + i) = i;
  }
  res_mult = skalar(var1, var2);
  printf("result multi = %f\n", res_mult);
  
  auto start_mono = std::chrono::high_resolution_clock::now();
  res_mono = skalar_mono(var1, var2);
  auto end_mono = std::chrono::high_resolution_clock::now();
  auto duration_mono = std::chrono::duration_cast<std::chrono::microseconds>(end_mono - start_mono);
  printf("result mono = %f\n", res_mono);
  std::cout << "time duration mono = " << duration_mono.count() << std::endl;
}

double* matrix_vector_multiplication_mono (double** matr, double* vec, int dim){
  double* result = new double[dim];
  for (int i = 0; i < dim; i++)
    *(result + i) = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      *(result + i) += *(*(matr + i) + j)*(*(vec + j));
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "time duration mono = " << duration.count() << std::endl;
  return result;
}

double* matrix_vector_multiplication_mult (double** matr, double* vec, int dim){
  double* result = new double[dim];
  for (int i = 0; i < dim; i++)
    *(result + i) = 0;
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      *(result + i) += *(*(matr + i) + j)*(*(vec + j));
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "time duration mult = " << duration.count() << std::endl;
  return result;
}//из тестов - многопоточное умножение матрицы на вектор быстрее однопоточного примерно в 3-5 раз

void matrix_vector_multiplication_test(){
  double **matrix = new double*[demension];
  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    *(matrix+i) = new double[demension];
  }

  double vec[demension];
  for (int i = 0; i < demension; i++){
    *(vec+i) = i%10;
    for (int j = 0; j < demension; j++){
      *(*(matrix + i) + j) = (j + i*4 + 1)%100;
    }
  }

  double* result = new double[demension];
  double* result_mult = new double[demension];
  result = matrix_vector_multiplication_mono(matrix, vec, demension);
  
  double sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (size_t i = 0; i < demension; i++)
  {
    sum += *(result + i);
  }
  std::cout<< "control sum for mono = " <<sum << std::endl;

  result_mult = matrix_vector_multiplication_mult(matrix, vec, demension);
  sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (size_t i = 0; i < demension; i++)
  {
    sum += *(result_mult + i);
  }
  std::cout<< "control sum for mult = " <<sum << std::endl;

  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    delete[] matrix[i];
  }
  delete[] matrix;
  delete[] result;
  delete[] result_mult;
}

int main()
{
  matrix_vector_multiplication_test();
}