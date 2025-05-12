#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>

#define demension 2000
std::string line;

void print_matr(double** A) {
    if(demension <= 24) {
      for(int i = 0; i < demension; i++){
        for(int j = 0; j < demension; j++){
         std::cout << A[i][j] << "\t";
        }
       std::cout << std::endl;
      }
    }
    std::cout << std::endl;

  }

void make_rand_sym_positive_matr(double** result) {//более простая версия
    omp_set_num_threads(12);
    double** A = new double*[demension];
    #pragma omp parallel for
    for (int i = 0; i < demension; i++){
      *(A + i) = new double[demension];
    }
  std::cout << "A crated" << std::endl;
    #pragma omp parallel 
    {
      thread_local std::mt19937 gen(std::random_device{}());
    #pragma omp for
    for(int i = 0; i < demension; i++){
      for(int j = 0; j < demension; j++){
        if(gen()%1000==0){//!
          *(*(A + i) + j) = ((unsigned int)gen())%100;
        }
        else {
          *(*(A + i) + j) = 0;
        }
        //*(*(A + j) + i) = *(*(A + i) + j);
      }
    }
  }
  
  std::cout << "A filled" << std::endl;
  
  double sum = 0;
  
  print_matr(A);

    #pragma omp parallel for private(sum)//schedule(static, 834) collapse(2)//коллапс делает из двух циклов один большой с i и j одновремнно. Почему размер чанка 100? Лучше брать chunk_size = N / (8 * число_потоков)
    for(int i = 0; i < demension; i++){//без коллапса у нас паралеллится только внешний цикл по i(j внутри каждого потока выполняется последовательно), а с ним по i и по j
      for(int j = 0; j < demension; j++){
        sum = 0;
        for(int k = 0; k < demension; k++){
          sum += A[i][k]*A[j][k]; //A*A^T
        }
        result[i][j] = sum;
      }
    }

    double k = rand()%100;
  
    #pragma omp parallel for
    for (int i = 0; i < demension; i++){
      result[i][i] += k; //диагональное смещение
    }
  
    print_matr(result);

    std::ofstream out[12];
    #pragma omp parallel for
    for (int i = 0; i < 12; i++){
        (*(out + i)).open("matrix_" + std::to_string(omp_get_thread_num()));
    }

    #pragma omp parallel for
    for(int i = 0; i < demension; i++){
        for(int j = 0; j < demension; j++){
            if(result[i][j] != 0){
                (*(out + omp_get_thread_num())) << i << " " << j << " " << result[i][j] << "\n"; //хранение в файле (x1, y1, val1, x2, y2, val2...)
            }
        }
    }

    for (int i = 0; i < 12; i++){
        (*(out + i)).seekp(0, std::ios::end);
        (*(out + i)) << EOF;
        (*(out + i)).close();
    }

    //теперь попробуем вывести...
    std::ifstream in[12];
    #pragma omp parallel for
    for (int i = 0; i < 12; i++){
        (*(in + i)).open("matrix_" + std::to_string(omp_get_thread_num()));
    }

    int file_num = 0;
    int x;
    int y;
    int peek;
    double val;
    char tmp;

    *(in + file_num) >> x >> y >> val;

    // for(int i = 0; i < demension; i++){
    //     for(int j = 0; j < demension; j++){
    //         if(file_num < 12 && i == x && j == y) {
    //             std::cout << val << "\t";
    //             //*(in + file_num) >> tmp;
    //             // if((*(in + file_num)).eof()) {
    //             //     (*(in + file_num)).close();
    //             //     file_num++;
    //             // }
    //             if (!(*(in + file_num) >> x >> y >> val)){
    //                 (*(in + file_num)).close();
    //                 file_num++;
    //                 if(file_num < 12){
    //                     (*(in + file_num)) >> x >> y >> val;
    //                 }
    //             }
    //         }
    //         else{
    //             std::cout<< 0 << "\t";
    //         }
    //     }
    //     std::cout<<std::endl;
    // }

    //(*(in + file_num)).close();

    std::cout << "matrix created and set in files successfully" << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < demension; i++){
      delete[] *(A + i);
    }
    delete[] A;
  }

  double norma (double* vector) {
    double* tmp = new double[omp_get_num_threads()];
    for(int i = 0; i < omp_get_num_threads(); i++){
        tmp[i] = 0;
    }
    #pragma omp parallel for
    for(int i = 0; i < demension; i++){
        tmp[omp_get_thread_num()] += vector[i]*vector[i];
    }
    double sum = 0;
    for(int i = 0; i < 12; i++){
        sum += tmp[i];
    }

    delete[] tmp;

    return sqrt(sum);
  }

  double* matrix_vector_multiplication_mult (double** matr, double* vec){
    double* result = new double[demension];
    for (int i = 0; i < demension; i++)
      *(result + i) = 0;

    #pragma omp parallel for
    for(int i = 0; i < demension; i++){
      for(int j = 0; j < demension; j++){
        *(result + i) += *(*(matr + i) + j)*(*(vec + j));
      }
    }
    return result;
  }

  double* matrix_vector_multiplication_mult_razr (double* vec){
    double* result = new double[demension];
    for (int i = 0; i < demension; i++)
      *(result + i) = 0;

    std::ifstream in[12];
    #pragma omp parallel for
    for (int i = 0; i < 12; i++){
        (*(in + i)).open("matrix_" + std::to_string(omp_get_thread_num()));
    }


    

    // for(int i = 0; i < 4; i++){
    //     while(in[i] >> x >> y >> val){
    //         std::cout<<val<<std::endl;
    //     }
    // }

    #pragma omp parallel// private(sum, x, x_prev, y, val)
    {
    if(demension < 12){
        omp_set_num_threads(demension);
    }
    int x; //в результирующем векторе это номер элемента (номер строки)
    int x_prev = 0; // для проверки на смену строки
    int y; //в векторе на который умножаем матрицу это номер элемента(строки)
    double val;
    int count = 0;
    int sum = 0;
    #pragma omp for
    for(int i = 0; i < omp_get_num_threads(); i++ ){//отдельный поток - отдельный файл
        x_prev = i*(demension/omp_get_num_threads());//небольшая оптимизация - чтобы на первой итерации while потоки не мешали друг другу
        while((*(in + omp_get_thread_num()) >> x >> y >> val)){
            if (x_prev != x && count != 0){
                *(result + x_prev) += sum;
                sum = 0;
            }
            sum += val*vec[y];
            x_prev = x;
            count++;
        }
        *(result + x_prev) = sum;
    }
}

    // for (int i = 0; i < demension; i++){
    //     std::cout << result[i] << std::endl;
    // }

    for (int i = 0; i < 12; i++){
        (*(in + i)).close();
    }

    return result;
  }

int main()
{
    omp_set_num_threads(12);
    int test[144];

    #pragma omp parallel for
    for (int i = 0; i < 144; i++){
        test[i] = omp_get_thread_num();
    }

    for (int i = 0; i < 144; i++){
        std::cout<< test[i] << std::endl;
    }
    double** preA = new double*[demension];
  #pragma omp parallel for
  for (int i= 0; i < demension; i++) {
    *(preA+i) = new double[demension];
  }

    make_rand_sym_positive_matr(preA);

    double* vector = new double[demension];
    double* result = new double[demension];

    for (int i = 0; i < demension; i++){
        *(vector + i) = i;
    }

    for (int i = 0; i < demension; i++){
        delete[] *(preA + i);
      }
      auto start = std::chrono::high_resolution_clock::now();
      result = matrix_vector_multiplication_mult_razr(vector);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cout << "time duration mult_razr = " << duration.count() << std::endl;


      auto start_m = std::chrono::high_resolution_clock::now();
      result = matrix_vector_multiplication_mult(preA, vector);
      auto end_m = std::chrono::high_resolution_clock::now();
      auto duration_m = std::chrono::duration_cast<std::chrono::microseconds>(end_m - start_m);
      std::cout << "time duration mult = " << duration_m.count() << std::endl;
    delete[] preA;
    delete[] vector;
    delete[] result;
    // double* test = new double[6];

    // for(int i = 0 ; i < 6; i++){
    //     test[i] = i;
    // }

    // std::cout << norma(test) << std::endl;

    // delete[] test;
    return 0;
}