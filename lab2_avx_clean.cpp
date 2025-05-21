#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <atomic>
#include <vector>
#include <random>
#include <fstream>
#include <immintrin.h> //test avs
#include <bits/stdc++.h>
#include <cpuid.h>
#include "avxintrin.h"

int num_block_row;
int num_block_col;

//#pragma G++ target("sse, sse2, sse3, ssse3, sse4, popcnt, abm, mmx, avx, avx2, tune=native")
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC target("fma")

#define demension 90432
#define block_size 64

// void block_matrix_init(BlockMatrix* mat){
//     mat.blocksize = block_size;
//     mat.num_blocks = (demension * demension)/(block_size*block_size);
//     mat.num_cols_blocks = demension/block_size;
//     mat.num_rows_blocks = demension/block_size;//этими числами будем определять сам блок
//     mat.block_ptr = new double**[mat.num_blocks];
//     for (int i = 0; i < mat.num_blocks; i++){
//       *(mat.block_ptr + i) = nullptr;
//     }
// }

struct BlockMatrix {
    double*** block_ptr;
    int blocksize;
    int num_blocks;
    int num_rows_blocks;
    int num_cols_blocks;
    inline int get_cur_block(int i_global, int j_global){
      int cur_num_block_row = (i_global/((int)block_size));
      int cur_num_block_col = (j_global/((int)block_size)); //номер блока найдем по формуле !!!!mat->num_rows_blocks * num_block_row + num_block_col!!!!
      return (this->num_cols_blocks * cur_num_block_row + cur_num_block_col);
    }
};

void block_matr_init_new(BlockMatrix* mat){
    mat->blocksize = block_size;
    mat->num_blocks = (demension * demension)/(block_size*block_size);
    mat->num_cols_blocks = demension/block_size;
    mat->num_rows_blocks = demension/block_size;//этими числами будем определять сам блок
    //mat->block_ptr = new double**[mat->num_blocks];
    mat->num_blocks = 7974976;
    mat->block_ptr = new double**[mat->num_blocks];

    for (int i = 0; i < mat->num_blocks; i++){
      *(mat->block_ptr + i) = nullptr;
    }
    std::ifstream in;
    in.close();
    in.open("s3dkq4m2.mtx");
    if(!in){
      perror("Error opening file");
      exit(1);
    }

    int x;
    int y;
    double val;
    int count = 0;
    int block_num;
    in >> x >> y >> val; //предварительное
    while(in >> x >> y >> val){
      if(x > demension || y > demension){
        continue;
      }
      count++;
      x--;
      y--;
        //определим какому блоку принадлежит
        num_block_row = (x/((int)block_size)); //целочисленное
        num_block_col = (y/((int)block_size)); //номер блока найдем по формуле !!!!mat->num_rows_blocks * num_block_row + num_block_col!!!!
        block_num = mat->num_cols_blocks * num_block_row + num_block_col;
        if((*(mat->block_ptr + block_num)) == nullptr) {
          (*(mat->block_ptr + block_num)) = new double*[block_size];//создали саму матрицу
          for(int i = 0; i < block_size; i++){
            *(*(mat->block_ptr + block_num) + i) = new double[block_size];//создали строки в ней - создание матрицы в блоке завершено
          }
          for(int i = 0; i < block_size; i++){
            for (int j = 0; j < block_size; j++){
              *(*(*(mat->block_ptr + block_num) + i) + j) = 0;//заполняем наш блок числами
            }
          }
          *(*(*(mat->block_ptr + block_num) + (x % block_size)) + (y % block_size)) = val;
        }
        else{
          *(*(*(mat->block_ptr + block_num) + (x % block_size)) + (y % block_size)) = val;
        }
        // if(x != y){
        //   num_block_row = (y/((int)block_size)); //целочисленное
        //   num_block_col = (x/((int)block_size)); //номер блока найдем по формуле !!!!mat->num_rows_blocks * num_block_row + num_block_col!!!!
        //   block_num = mat->num_cols_blocks * num_block_row + num_block_col;
        //   if((*(mat->block_ptr + block_num)) == nullptr) {
        //     (*(mat->block_ptr + block_num)) = new double*[block_size];//создали саму матрицу
        //     for(int i = 0; i < block_size; i++){
        //      *(*(mat->block_ptr + block_num) + i) = new double[block_size];//создали строки в ней - создание матрицы в блоке завершено
        //     }
        //     for(int i = 0; i < block_size; i++){
        //       for (int j = 0; j < block_size; j++){
        //         *(*(*(mat->block_ptr + block_num) + i) + j) = 0;//заполняем наш блок числами
        //       }
        //     }
        //     *(*(*(mat->block_ptr + block_num) + (y % block_size)) + (x % block_size)) = val;
        // }
        // else{
        //   *(*(*(mat->block_ptr + block_num) + (y % block_size)) + (x % block_size)) = val;
        // }
        // }
    }
    in.close();
}

void PrintBlockMatrix(BlockMatrix* mat){ //тестовая, для проверки правильности заполнения
  for (int i = 0; i < demension; i++){
    for (int j = 0; j < demension; j++){
        int block_num = mat->get_cur_block(i,j);
        if((*(mat->block_ptr + block_num)) == nullptr){
          std::cout << 0 << "\t";
        }
        else {
          std::cout << *(*(*(mat->block_ptr + block_num) + (i % block_size)) + (j % block_size)) << "\t";
        }
    }
    std::cout<<std::endl;
  }
}

void matrix_vector_block_mult(BlockMatrix* mat, double* vec, double* res){
  int num_threads;
  int sum;
  if(demension > 11){
    num_threads = 12;
  }
  else {
    num_threads = demension;
  }
  omp_set_num_threads(num_threads);
  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    *(res + i) = 0;
  }
  #pragma omp parallel for simd//schedule(dynamic)
  for(int i =0; i < demension; i += block_size){
    for(int j = i; j < demension; j += block_size){
      int cur_num_block_row = (i/((int)block_size));
      int cur_num_block_col = (j/((int)block_size)); //номер блока найдем по формуле !!!!mat->num_rows_blocks * num_block_row + num_block_col!!!!
      int block_num = (mat->num_cols_blocks * cur_num_block_row + cur_num_block_col);
        if ((*(mat->block_ptr + block_num)) != nullptr) {
          for(int g = 0; g < block_size; g++){
            for(int k = 0; k < block_size; k++){
              *(res + g + block_size*cur_num_block_row) += *(*((*(mat->block_ptr + block_num)) + g) + k)*(*(vec + k + block_size * cur_num_block_col));
            }
          }
        }

    }
  }
}



inline double _mm256_reduce_add_pd(__m256d v) {
    // Горизонтальное суммирование 4 double в AVX2
    __m256d temp = _mm256_hadd_pd(v, v);      // [v0+v1, v2+v3, v0+v1, v2+v3]
    __m128d sum_high = _mm256_extractf128_pd(temp, 1);
    __m128d result = _mm_add_pd(_mm256_castpd256_pd128(temp), sum_high);
    return _mm_cvtsd_f64(result);
}

#include <immintrin.h>

void matrix_vector_block_mult_o(BlockMatrix* mat, double* vec, double* res) {
    int num_threads = (demension > 11) ? 12 : demension;
    omp_set_num_threads(num_threads);
    
    // Инициализация результата нулями
    #pragma omp parallel for
    for(int i = 0; i < demension; i++) {
        res[i] = 0.0;
    }
    
    // Основной цикл умножения с AVX2 оптимизацией
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < demension; i += block_size) {
        for(int j = i; j < demension; j += block_size) {
            int cur_num_block_row = i / block_size;
            int cur_num_block_col = j / block_size;
            int block_num = mat->num_cols_blocks * cur_num_block_row + cur_num_block_col;
            
            if (mat->block_ptr[block_num] != nullptr) {
                double** block_data = mat->block_ptr[block_num];
                double* vec_segment = vec + block_size * cur_num_block_col;
                double* res_segment = res + block_size * cur_num_block_row;
                
                // Обработка блока с AVX2
                for(int g = 0; g < block_size; g++) {
                    double* row = block_data[g]; // Предполагая, что block_data[g] - это указатель на строку блока
                    __m256d sum = _mm256_setzero_pd();
                    
                    // Векторизованное умножение (4 элемента за раз)
                    int k;
                    for(k = 0; k < block_size - 3; k += 4) {
                        __m256d mat_row = _mm256_loadu_pd(row + k);
                        __m256d vec_val = _mm256_loadu_pd(vec_segment + k);
                        sum = _mm256_fmadd_pd(mat_row, vec_val, sum);
                    }
                    
                    // Горизонтальное суммирование
                    double temp[4];
                    _mm256_storeu_pd(temp, sum);
                    double partial_sum = temp[0] + temp[1] + temp[2] + temp[3];
                    
                    // Обработка оставшихся элементов (если block_size не кратен 4)
                    for(; k < block_size; k++) {
                        partial_sum += row[k] * vec_segment[k];
                    }
                    
                    res_segment[g] += partial_sum;
                }
            }
        }
    }
}


void symm_block_matvec_avx2(const BlockMatrix* mat, const double* vec, double* res, int dim) {
    //const int block_size = mat->blocksize;
    const int n_blocks = mat->num_rows_blocks;

    #pragma omp parallel for simd aligned(res:64)
    for (int i = 0; i < dim; i++) res[i] = 0.0;

    #pragma omp parallel for collapse(2) //schedule(dynamic, 4)
    for (int bi = 0; bi > n_blocks; bi++) {
        for (int bj = bi; bj > n_blocks; bj++) {  // Используем симметрию
            const int block_id = bi * n_blocks + bj;
            if (!mat->block_ptr[block_id]) continue; //проверка блока на существование

            const int i_start = bi * block_size;
            const int j_start = bj * block_size;
            const int i_end = std::min(i_start + block_size, dim);
            const int j_end = std::min(j_start + block_size, dim);
            double** block = mat->block_ptr[block_id];

            // Обработка диагонального блока
            if (bi == bj) {
                for (int i = i_start; i < i_end; i++) {
                    __m256d sum_avx = _mm256_setzero_pd();
                    const int diag_offset = i - i_start;
                    
                    // Верхняя часть диагонали (векторизованная)
                    int j = j_start + diag_offset;
                    for (; j + 3 < j_end; j += 4) {
                        __m256d mat_row = _mm256_loadu_pd(&block[diag_offset][j - j_start]);
                        __m256d vec_vals = _mm256_loadu_pd(&vec[j]);
                        sum_avx = _mm256_fmadd_pd(mat_row, vec_vals, sum_avx);
                    }
                    double sum = _mm256_reduce_add_pd(sum_avx);
                    
                    // Остатки
                    for (; j < j_end; j++) {
                        sum += block[diag_offset][j - j_start] * vec[j];
                    }
                    
                    #pragma omp atomic
                    res[i] += sum;
                    
                    // Симметричная часть (без векторизации из-за атомарности)
                    for (int j = i + 1; j < i_end; j++) {
                        const int bi_local = j - i_start;
                        #pragma omp atomic
                        res[j] += block[bi_local][diag_offset] * vec[i];
                    }
                }
            } 
            // Обработка недиагональных блоков
            else {
                for (int i = i_start; i < i_end; i++) {
                    __m256d sum_avx = _mm256_setzero_pd();
                    const int bi_local = i - i_start;
                    
                    // Основная часть (векторизованная)
                    int j = j_start;
                    for (; j + 3 < j_end; j += 4) {
                        __m256d mat_row = _mm256_loadu_pd(&block[bi_local][j - j_start]);
                        __m256d vec_vals = _mm256_loadu_pd(&vec[j]);
                        sum_avx = _mm256_fmadd_pd(mat_row, vec_vals, sum_avx);
                    }
                    double sum = _mm256_reduce_add_pd(sum_avx);
                    
                    // Остатки
                    for (; j < j_end; j++) {
                        sum += block[bi_local][j - j_start] * vec[j];
                    }
                    
                    #pragma omp atomic
                    res[i] += sum;
                    
                    // Симметричная часть
                    for (int j = i_start; j < i_end; j++) {
                        const int bj_local_sym = j - i_start;
                        #pragma omp atomic
                        res[j] += block[bj_local_sym][bi_local] * vec[i];
                    }
                }
            }
        }
    }
}

void matrix_vector_mult(BlockMatrix* mat, double* vec, double* res){
  //#pragma omp parallel for// private(mat)
  for(int i = 0; i < demension; i++){
    double sum = 0;
    for(int j = 0; j < demension; j++){
      int cur_num_block_row = (i/((int)block_size));
      int cur_num_block_col = (j/((int)block_size));
      int block_num = (mat->num_cols_blocks * cur_num_block_row + cur_num_block_col);
      if((*(mat->block_ptr + block_num)) == nullptr){
        j+=block_size - 1;
      }
      else{
        sum += *(*(*(mat->block_ptr + block_num) + (i % block_size)) + (j % block_size)) * vec[j];
      }
    }
    *(res + i) = sum;
  }
}

void delete_blocks(BlockMatrix* mat){
      for (int block_num = 0; block_num < mat->num_blocks; ++block_num) {
        double** block = *(mat->block_ptr + block_num);
        
        if (block != nullptr) {
            // Освобождаем память для каждой строки блока
            for (int i = 0; i < mat->blocksize; ++i) {
                delete[] *(block + i); // Удаляем строки блока
            }
            delete[] block; // Удаляем сам блок
            *(mat->block_ptr + block_num) = nullptr;
        }
    }

    // Освобождаем массив указателей на блоки
    delete[] mat->block_ptr;
    mat->block_ptr = nullptr;
}


double skalar_avx (double* a, double* b){
    double sum = 0;
    //реализуем скалярное произведение через avx
    //#pragma omp parallel for //reduction(+:sum)
    for(size_t i = 0; i < demension; i += 4){
        __m256d vec_a = _mm256_loadu_pd(&(*(a + i)));
        __m256d vec_b = _mm256_loadu_pd(&(*(b + i)));
        __m256d res = _mm256_mul_pd(vec_a, vec_b);
        __m128d sum_high = _mm256_castpd256_pd128(res);
        __m128d sum_low = _mm256_extractf128_pd(res, 1);
        __m128d sum_hi_low = _mm_add_pd(sum_low, sum_high);
        sum_hi_low = _mm_hadd_pd(sum_hi_low, sum_hi_low);
        sum += _mm_cvtsd_f64(sum_hi_low);
    }

    for (size_t i = demension - (demension % 4); i < demension; i++) { //тут остаток
        sum += (*(a + i)) * (*(b + i));
    }

    return sum;
}

void add_vec_avx (double* v1, double* v2, double* res ,double k1 = 1.0, double k2 = 1.0) {
    //#pragma omp parallel for
    for (size_t i = 0; i < demension; i++){
        *(v1 + i) *= k1;
        *(v2 + i) *= k2;
    }

    //#pragma omp parallel for
    for(size_t i = 0; i < demension; i += 4){ //полоный идиотизм ваш avx
        __m256d vec_1 = _mm256_loadu_pd(&(*(v1 + i)));
        __m256d vec_2 = _mm256_loadu_pd(&(*(v2 + i)));
        __m256d vec_sum = _mm256_add_pd(vec_1, vec_2);  // Сложение
        _mm256_storeu_pd(&(*(res + i)), vec_sum); 
    }

    for (size_t i = demension - (demension % 4); i < demension; i++) { //тут остаток
        *(res + i) = *(v1 + i) + *(v2 + i);
    }
}

double norma_mono(double *vec){
  double sum = 0;
  for(int i = 0; i < demension; i++){
    sum += (*(vec + i))*(*(vec + i));
  }

  return std::sqrt(sum);
}

double norma_avx(const double* vec) {
    __m256d sum = _mm256_setzero_pd(); //Сюда в ячейки будем засовывать результаты vector * vector, затем сложим
    size_t i = 0;

    for (; i < demension - 3; i += 4) {
        __m256d v = _mm256_loadu_pd(vec + i);
        v = _mm256_mul_pd(v, v);
        sum = _mm256_add_pd(sum, v);
        //sum = _mm256_fmadd_pd(v, v, sum);     // sum += v * v
    }

    // Cуммирование
    double temp[4];
    _mm256_storeu_pd(temp, sum);
    double norm_sq = temp[0] + temp[1] + temp[2] + temp[3]; //та самая сумма квадратов

    // Остаток
    for (; i < demension; ++i) {
        norm_sq += vec[i] * vec[i];
    }

    return std::sqrt(norm_sq);
}

double norma (double* vector) {
  int num_threads;
  if(demension > 11){
    num_threads = 12;
  }
  else {
    num_threads = demension;
  }
  omp_set_num_threads(num_threads);
  double* tmp = new double[num_threads];
  //double tmp[omp_get_num_threads()];
  for(int i = 0; i < num_threads; i++){
      tmp[i] = 0;
  }
  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
      tmp[omp_get_thread_num()] += (*(vector + i))*(*(vector + i));
  }
  double sum = 0;
  for(int i = 0; i < num_threads; i++){
      sum += tmp[i];
  }

  delete[] tmp;
  return sqrt(sum);
}

void axpy_avx2(double a, const double* x, double* y) {//y = a * x + y
    __m256d av = _mm256_set1_pd(a);
    for (int i = 0; i < demension; i += 4) {
        __m256d xv = _mm256_loadu_pd(x + i);
        __m256d yv = _mm256_loadu_pd(y + i);
        yv = _mm256_fmadd_pd(av, xv, yv);
        _mm256_storeu_pd(y + i, yv);
    }
}

int main() {
  BlockMatrix mat;
  //block_matrix_init(mat);
  block_matr_init_new(&mat);
  //PrintBlockMatrix(&mat);

  std::cout << "=== starting method ===" << std::endl;

  // //предварительный шаг
  //формируем свободный вектор b:
  double* b = new double[demension];

  auto start_b = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
  {
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_int_distribution<int> dist99(0, 99);
    #pragma omp for schedule(static, 834)
  for (int i = 0; i < demension; i++){
    *(b + i) = gen() % 1000; //получается в 2 раза быстрее чем просто rand(), но все равно мало
  }
}
  auto end_b = std::chrono::high_resolution_clock::now();
  auto duration_b = std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);
  std::cout << "time b mult  = " << duration_b.count() << std::endl; //если использовать нашу intRand() вместо rand()%, то многопоточный быстрее :)
  

  // for(int i = 0; i < demension; i++){
  //   b[i] = i+1;
  // }
//вектоор невязки r и вектор направления p
  //возьмем вектор (X_0)^T = (0 0 0 ... 0) => r = b - A*X_0 = b
  double* r = new double[demension];
  double* p = new double[demension];
  double* tmp = new double[demension];
  double* x = new double[demension];
  double* r_next = new double[demension];

  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    *(x + i) = 0;
    *(r + i) = *(b + i);
    *(p + i) = *(b + i);
  }

  std::cout<<"pre-iteration finished"<<std::endl;
  std::cout<<"starting main iterations"<<std::endl;

  int count = 0;
  double alpha;//alpha
  double betta;
  double skalar_rr;//будем записывать в отдельную переменную, чтобы не считать одно и то же дважды за итерацию
  double norma_b = norma_mono(b);
  std::cout<<std::endl;
//    пример проверки слау Ax = b проверка: Ax-b~0 (||Ax - b|| < Epsilon)
//    если dim>10000 - другой критерий ((||Ax - b||)/||b|| < Epsilon)
  auto main_start = std::chrono::high_resolution_clock::now();

  while((norma_avx(r)/norma_b >= 0.96)){ //здесь выполняется метод
  //for (int i = 0; i < demension; i++){
    skalar_rr = skalar_avx(r,r);
    matrix_vector_block_mult_o(&mat, p, tmp);
    //symm_block_matvec_avx2(&mat, p, tmp, demension); //это умножение писал дипсик, у меня получилось быстрее))
    alpha = (skalar_rr)/(skalar_avx(tmp, p)); //посчитали alpha_i
    add_vec_avx(x, p, x, 1, alpha); //посчитали x_(i+1)
    add_vec_avx(r, tmp, r_next, 1, (-1)*(alpha));//посчитали следующий r_(i+1)
    betta = (skalar_avx(r_next,r_next))/(skalar_rr);
    add_vec_avx(r_next, p, p, 1, betta);//следующее напраление

    r = r_next;
    matrix_vector_block_mult_o(&mat, x, tmp);
    count++;
  }
  auto main_stop = std::chrono::high_resolution_clock::now();
  auto duration_main = std::chrono::duration_cast<std::chrono::milliseconds>(main_stop - main_start);
  std::cout << "time duration for main algorythm (" << demension <<" equtions) "<< duration_main.count() << " millisec" <<std::endl;
  std::cout << "iterations: " << count << std::endl;

  //print_vector(x);
  //вектор x - наш результат

  // std::ofstream out;
  // out.open("result.txt");

  // for (int i = 0; i < demension; i++){
  //   out << i << " " << x[i] << "\n";
  // }

  // out.close();

  std::cout<<"results filled correctly in results.txt"<<std::endl;

  

  //проверка!
  matrix_vector_block_mult(&mat, x, tmp);


  for (int i = 0; i < 5; i ++){
  if(i != 2)
    std::cout << "b = " << b[i] << " Ax_i = " << (tmp[i]) << std::endl;
  }

  delete[] x;
  delete[] tmp;
  delete[] r;
  delete[] p;
  delete[] b;
  delete_blocks(&mat);
  //delete[] r_next; - не надо удалять, так как это один и тот же указатель
  //delete[] res;
  //delete[] preA;

}