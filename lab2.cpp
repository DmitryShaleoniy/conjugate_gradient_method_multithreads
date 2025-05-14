#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <atomic>
#include <vector>
#include <random>
#include <fstream>

#define demension 10000
#define epsilon 0.0000001

int intRand(const int & min, const int & max) {
  static thread_local std::mt19937 generator;
  std::uniform_int_distribution<int> distribution(min,max);
  return distribution(generator);
} //этот генератор безопасен для многопоточного использования, т.к. rand() - это общий ресурс на всех, 
//а здесь мы создаём отдельный генератор для каждого потока


void print_matr(double** A) {
  if(demension <= 10) {
    for(int i = 0; i < demension; i++){
      for(int j = 0; j < demension; j++){
       std::cout << A[i][j] << "\t";
      }
     std::cout << std::endl;
    }
  }
}

void print_vector(double* vec){
  if(demension <=10) {
    for(int i = 0; i < demension; i++){
      std::cout<<vec[i]<<std::endl;
    }
  }
}

//функция вычисления скалярного произведения 
double skalar (double *var1, double* var2) {//8 потоков5
  omp_set_num_threads(12);
  double res = 0;
  double th_tmp[12] {0};
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(static, 10)
  for(int i = 0; i < demension; i++) {
    // if(i == 0 || i == demension/2 || i == demension*3/4) {
    //   printf("thread num = %d\n", omp_get_thread_num());
    // }
    th_tmp[omp_get_thread_num()] += (*(var1 + i))*(*(var2 + i));
  }
#pragma omp parallel for reduction(+:res)
for(int i = 0; i < 12; i++){
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
    // if(i == 0 || i == demension/2 || i == demension*3/4) {
    //   printf("thread num = %d\n", omp_get_thread_num());
    // }
    res += (*(var1 + i))*(*(var2 + i));
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "time duration multi = " << duration.count() << std::endl;
  return res;
}//работа ядер сводится к ожиданию

double skalar_mono (double *var1, double* var2) { //в одном потоке
  double res = 0;
  auto start_mono = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < demension; i++) {
    // if(i == 0 || i == demension/2 || i == demension*3/4) {
    //   printf("thread num = %d\n", omp_get_thread_num());
    // }
    res += (var1[i] * var2[i]);
  }
  auto end_mono = std::chrono::high_resolution_clock::now();
  auto duration_mono = std::chrono::duration_cast<std::chrono::microseconds>(end_mono - start_mono);
  //std::cout << "time duration mono = " << duration_mono.count() << std::endl;
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
 
  res_mono = skalar_mono(var1, var2);
  printf("result mono = %f\n", res_mono);
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

double* matrix_vector_multiplication_mult (double** matr, double* vec){
  double* result = new double[demension];
  for (int i = 0; i < demension; i++)
    *(result + i) = 0;
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(int i = 0; i < demension; i++){
    for(int j = 0; j < demension; j++){
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

  result_mult = matrix_vector_multiplication_mult(matrix, vec);
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

void matrix_matrix_mult(double** m1, double** m2, double** result){
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for collapse(2) schedule(dynamic, 1000)
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

void make_rand_sym_semi_positive_matr (double** res){//альтернативная версия
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

  #pragma omp parallel
  {
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_int_distribution<int> dist99(0, 99); //создаём 1 раз на поток
    thread_local std::uniform_int_distribution<int> dist9(0, 9);
    thread_local std::uniform_int_distribution<int> dist999(0, 999);

  #pragma omp for
  for(int i = 0; i < demension; i++){
    for(int j = i; j < demension; j++){
      if(gen()%10==0){
        *(*(V + i) + j) = gen()%100;
      }
      else {
        *(*(V + i) + j) = 0;
      }
      *(*(A + i) + j) = 0;
      *(*(V + j) + i) = *(*(V + i) + j);
    }
    if(dist999(gen)==0){
      *(*(A + i) + i) = (double)dist9(gen);
    }
  }
}
  
  // for (int i = 0; i < demension; i++){
  //   for (int j = 0; j < demension; j++){
  //     std::cout << V[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  
  matrix_matrix_mult(V,A,tmp);
  matrix_matrix_mult(tmp,V,res); //обусловимся, что V задали симметрической

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

void make_rand_sym_positive_matr(double** result) {//главная версия
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

    for(int i = 0; i < demension; i++){
        for(int j = 0; j < demension; j++){
            if(file_num < 12 && i == x && j == y) {
                std::cout << val << "\t";
                //*(in + file_num) >> tmp;
                // if((*(in + file_num)).eof()) {
                //     (*(in + file_num)).close();
                //     file_num++;
                // }
                if (!(*(in + file_num) >> x >> y >> val)){
                    (*(in + file_num)).close();
                    file_num++;
                    if(file_num < 12){
                        (*(in + file_num)) >> x >> y >> val;
                    }
                }
            }
            else{
                std::cout<< 0 << "\t";
            }
        }
        std::cout<<std::endl;
    }

    //(*(in + file_num)).close();

    #pragma omp parallel for
    for (int i = 0; i < demension; i++){
      delete[] *(A + i);
    }
    delete[] A;
}


//теперь сделаем эффективное хранение разреженных матриц
void make_optimized(std::vector<double> &rows, std::vector<double> &cols, std::vector<double> &vals){ //старая версия, без параллелизма
  double** A = new double*[demension];
  #pragma omp parallel for
  for (int i = 0; i < demension; i++){
    *(A + i) = new double[demension];
  }

  #pragma omp parallel 
  {
    thread_local std::mt19937 gen(std::random_device{}());
    //thread_local std::uniform_int_distribution<int> dist99(0, 99); //создаём 1 раз на поток
    //thread_local std::uniform_int_distribution<int> dist9(0, 9);
  #pragma omp for
  for(int i = 0; i < demension; i++){
    for(int j = i; j < demension; j++){
      if(gen()%10==0){
        *(*(A + i) + j) = gen()%100;
      }
      else {
        *(*(A + i) + j) = 0;
      }
      *(*(A + j) + i) = *(*(A + i) + j);
    }
  }
}

  double z = intRand(0,99);
  double tmp;

  print_matr(A);

  for(int i = 0; i < demension; i++){
    for(int j = 0; j < demension; j++){
      tmp = 0;
      for(int k = 0; k < demension; k++){//A*A^T
        tmp += A[i][k]*A[k][j];
      }
      if (tmp != 0){
        vals.push_back(tmp + ((double)(i == j))*z); //вместе с диагональным смещением
        rows.push_back(i);
        cols.push_back(j);
      }
      else if(i == j){
        vals.push_back(z); //вместе с диагональным смещением
        rows.push_back(i);
        cols.push_back(j);
      }
    }
  }

  vals.shrink_to_fit();
  rows.shrink_to_fit();
  cols.shrink_to_fit();

  #pragma omp parallel for
  for (int i = 0; i < demension; i++){
    delete[] *(A + i);
  }
  delete[] A;
}//заполнение в 5 раз медленне чем многопоточное заполнение, но полной матрицы

double** add_matr_mult (double** m1, double** m2, double** res ,double k1 = 1.0, double k2 = 1.0){ //классическое представление матриц
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < demension; i++){
    for (int j = 0; j < demension; j++){
      //res[i][j] = 0;
      res[i][j] = (k1*m1[i][j] + k2*m2[i][j]);
    }
  }
  return res;
}

double** add_matr_mono (double** m1, double** m2, double** res ,double k1 = 1.0, double k2 = 1.0){ //классическое представление матриц
  for (int i = 0; i < demension; i++){
    for (int j = 0; j < demension; j++){
      //res[i][j] = 0;
      res[i][j] = (k1*m1[i][j] + k2*m2[i][j]);
    }
  }
  return res;
}

void test_add_matr(){
  double** preA = new double*[demension];
  double** res = new double*[demension];

  #pragma omp parallel for
  for (int i= 0; i < demension; i++) {
    *(preA+i) = new double[demension];
    *(res + i) = new double[demension];
  }
  make_rand_sym_positive_matr(preA);

  auto start = std::chrono::high_resolution_clock::now();
  add_matr_mult(preA, preA, res);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "time duration mult = " << duration.count() << std::endl;

  auto start_m = std::chrono::high_resolution_clock::now();
  add_matr_mono(preA, preA, res);
  auto end_m = std::chrono::high_resolution_clock::now();
  auto duration_m = std::chrono::duration_cast<std::chrono::microseconds>(end_m - start_m);
  std::cout << "time duration mono = " << duration_m.count() << std::endl;

  #pragma omp parallel for
  for (int i = 0; i < demension; i++){
    delete[] *(preA + i);
    delete[] *(res + i);
  }
  delete[] res;
  delete[] preA;
} // многопоточная реализация в 2.5 раза быстрее (без разрежения)

double* add_vec_mult (double* v1, double* v2, double* res ,double k1 = 1.0, double k2 = 1.0){ //классическое представление матриц
  #pragma omp parallel for
  for (int i = 0; i < demension; i++){
      res[i]= (k1*v1[i] + k2*v2[i]);
  }
  return res;
}

double norma (double* vector) {
  double* tmp = new double[demension];
  //double tmp[omp_get_num_threads()];
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

void matrix_vector_multiplication_mult_razr (double* vec, double* result){//версия для разреженной матрицы при хранении в файлах
  //double* result = new double[demension];
  for (int i = 0; i < demension; i++)
    *(result + i) = 0;

  std::ifstream in[12];
  #pragma omp parallel for
  for (int i = 0; i < 12; i++){
      (*(in + i)).open("matrix_" + std::to_string(omp_get_thread_num()));
  }

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
  double sum = 0;
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

  //return result;
}


int main() {
  //skalar_test();

//    srand(time(0));
//   double** preA = new double*[demension];
//   double** res = new double*[demension];
//   #pragma omp parallel for
//   for (int i= 0; i < demension; i++) {
//     *(preA+i) = new double[demension];
//     *(res + i) = new double[demension];
//   }

//   auto start = std::chrono::high_resolution_clock::now();
//   //make_rand_sym_positive_matr(preA);
//   auto end = std::chrono::high_resolution_clock::now();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//   std::cout << "time duration mult = " << duration.count() << std::endl;

//   std::vector<double> rows; 
//   std::vector<double> cols;
//   std::vector<double> vals;

//   rows.reserve(demension * demension);
//   cols.reserve(demension * demension);
//   vals.reserve(demension * demension);

//   auto start_m = std::chrono::high_resolution_clock::now();
//   //make_optimized(rows, cols, vals);
//   auto end_m = std::chrono::high_resolution_clock::now();
//   auto duration_m = std::chrono::duration_cast<std::chrono::microseconds>(end_m - start_m);
//   std::cout << "time duration mono = " << duration_m.count() << std::endl;

//   int k = 0;

//   if(demension <= 10){
//   for (size_t i = 0; i < demension; i++){
//     for (size_t j = 0; j < demension; j++){
//         if(i == rows[k] && j == cols[k] && k < rows.size()){
//           std::cout<<vals[k]<<"\t";
//           k++;
//         }
//         else 
//           std::cout<< "0" <<"\t";
//     }
//     std::cout<<std::endl;
//   }
// }

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
    *(b + i) = gen() % 100; //получается в 2 раза быстрее чем просто rand(), но все равно мало
  }
}
  auto end_b = std::chrono::high_resolution_clock::now();
  auto duration_b = std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);
  std::cout << "time b mult  = " << duration_b.count() << std::endl; //если использовать нашу intRand() вместо rand()%, то многопоточный быстрее :)

  //ДЛЯ ТЕСТА!!!!!

  // *(b) = 6;
  // *(b + 1) = 6;
  // *(b + 2) = 8;
  // *(b + 3) = 4;

  // auto start_bm = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < demension; i++){
  //   *(b + i) = rand()%100; //для однопоточного rand() будет быстрее 
  // }
  // auto end_bm = std::chrono::high_resolution_clock::now();
  // auto duration_bm = std::chrono::duration_cast<std::chrono::microseconds>(end_bm - start_bm);
  // std::cout << "time b mono  = " << duration_bm.count() << std::endl;

  
//вектоор невязки r и вектор направления p
  auto main_start = std::chrono::high_resolution_clock::now();
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
  double norma_b = norma(b);

//    пример проверки слау Ax = b проверка: Ax-b~0 (||Ax - b|| < Epsilon)
//    если dim>10000 - другой критерий ((||Ax - b||)/||b|| < Epsilon)

  while((norma(r) >= 0.0001)){ //здесь выполняется метод
    skalar_rr = skalar_mono(r,r);
    matrix_vector_multiplication_mult_razr(p, tmp);//это A*p_i
    alpha = (skalar_rr)/(skalar_mono(tmp, p)); //посчитали alpha_i
    add_vec_mult(x, p, x, 1, alpha); //посчитали x_(i+1)
    add_vec_mult(r, tmp, r_next, 1, (-1)*(alpha));//посчитали следующий r_(i+1)
    betta = (skalar_mono(r_next,r_next))/(skalar_rr);
    add_vec_mult(r_next, p, p, 1, betta);

    r = r_next;
    matrix_vector_multiplication_mult_razr(x, tmp);//Ax
    count++;
  }
      printf("\n");


  print_vector(x);
  //вектор x - наш результат

  std::ofstream out;
  out.open("result.txt");

  for (int i = 0; i < demension; i++){
    out << i << " " << x[i] << "\n";
  }

  out.close();

  std::cout<<"results filled correctly in results.txt"<<std::endl;

  auto main_stop = std::chrono::high_resolution_clock::now();
  auto duration_main = std::chrono::duration_cast<std::chrono::seconds>(main_stop - main_start);
  std::cout << "time duration for main algorythm (" << demension <<" equtions)"<< duration_main.count() << " sec" <<std::endl;

  #pragma omp parallel for
  for (int i = 0; i < demension; i++){
    //delete[] *(preA + i);
    //delete[] *(res + i);
  }
  delete[] tmp;
  delete[] x;
  delete[] r;
  delete[] p;
  delete[] b;
  //delete[] r_next; - не надо удалять, так как это один и тот же указатель
  //delete[] res;
  //delete[] preA;
}