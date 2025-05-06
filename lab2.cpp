#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <atomic>

#define demension 1000
#define epsilon pow(10, -4)

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
  //time_t start_time, end_time, start_time_mono, end_time_mono;
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

void make_rand_sym_positive_matr (double** res){
  // VAV^T - V - матрица собственных векторов, А - диагональная матрица собственных значений
  srand(time(0));
  double** V = new double*[demension];
  double** A = new double*[demension];
  double** tmp = new double*[demension];
  #pragma omp parallel for
  for (int i = 0; i < demension; i++){
    *(tmp + i) = new double[demension];
    *(V + i) = new double[demension];
    *(A + i) = new double[demension];
  }

  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    for(int j = 0; j < demension; j++){
      if(rand()%10==0){
        *(*(V + i) + j) = rand()%100;
      }
      else {
        *(*(V + i) + j) = 0;
      }
      *(*(A + i) + j) = 0;
    }
    if(rand()%1000==0){
      *(*(A + i) + i) = rand()%10;
    }
  }

  
  for (int i = 0; i < demension; i++){
    
  }
  
  //matrix_matrix_mult(V,A,tmp);
  //matrix_matrix_mult(tmp,V,res); //обусловимся, что V задали симметрической

  #pragma omp parallel for
  for (int i = 0; i < demension; i++){
    delete[] *(tmp + i);
    delete[] *(V + i);
    delete[] *(A + i);
  }
  delete[] tmp;
  delete[] V;
  delete[] A;
}

void matrix_matrix_mult(double** m1, double** m2, double** result){
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    for(int j = 0; j < demension; j++){
      result[i][j] = 0;
      for(int k = 0; k < demension; k++){
        result[i][j] += m1[i][k]*m2[k][j];
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "time duration mult = " << duration.count() << std::endl;
}

void matrix_matrix_mono(double** m1, double** m2, double** result){
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < demension; i++){
    for(int j = 0; j < demension; j++){
      result[i][j] = 0;
      for(int k = 0; k < demension; k++){
        result[i][j] += m1[i][k]*m2[k][j];
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "time duration mono = " << duration.count() << std::endl;
}

void matr_mult_test(){
  srand(time(0));
  double **matrix1 = new double*[demension];
  double **matrix2 = new double*[demension];
  double **result = new double*[demension];
  double **result_mult = new double*[demension];

  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    *(matrix1+i) = new double[demension];
    *(matrix2+i) = new double[demension];
    *(result+i) = new double[demension];
    *(result_mult+i) = new double[demension];
  }

  for (size_t i = 0; i < demension; i++){
    for (size_t j = 0; j < demension; j++){
      *(*(matrix1 + i) + j) = rand()%100;
      *(*(matrix2 + i) + j) = rand()%100;
    }
  }
  

  matrix_matrix_mono(matrix1, matrix2, result);
  double sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < demension; i++){
    for (int j = 0; j < demension; j++){
      sum += *(*(result + i) + j);
    }
  }
  std::cout<< "control sum for mono = " <<sum << std::endl;

  matrix_matrix_mult(matrix1, matrix2, result_mult);

  sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < demension; i++){
    for (int j = 0; j < demension; j++){
      sum += *(*(result + i) + j);
    }
  }
  std::cout<< "control sum for mult = " <<sum << std::endl;

  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    delete[] matrix1[i];
    delete matrix2[i];
  }
  delete[] matrix1;
  delete[] matrix2;
  delete[] result;
  delete[] result_mult;
}

int main() {
  srand(time(0));
  // //реализуем алгоритм
  // //задать матрицу А - разряженная и симметрическая размерности demension
  // //double preA[demension][demension];
  // double** preA = new double*[demension];
  // #pragma omp parallel for
  // for (int i= 0; i < demension; i++) {
  //   *(preA+i) = new double[demension];
  // }
  // int counter = 0;
  // auto start = std::chrono::high_resolution_clock::now();
  // //#pragma omp parallel for
  // for (size_t i = 0; i < demension; i++){
  //   for (size_t j = i; j < demension; j++) {
  //     if(rand()%10==7){
  //       preA[i][j] = rand()%100;
  //     }
  //     else {
  //       preA[i][j] = 0;
  //     }
  //     preA[j][i] = preA[i][j];
  //   }
  // }
  // std::cout<<counter<<std::endl;
  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // std::cout << "time duration = " << duration.count() << std::endl;


  // #pragma omp parallel for
  // for (int i = 0; i < demension; i++){
  //   delete[] *(preA + i);
  // }
  // delete[] preA;
  // return 0;
  matr_mult_test();
}