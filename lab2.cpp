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

int main()
{
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