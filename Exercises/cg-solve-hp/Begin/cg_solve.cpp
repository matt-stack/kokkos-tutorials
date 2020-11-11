#include<cstdio>
#include<cstdlib>
#include<Kokkos_Core.hpp>
#include<generate_matrix.hpp>


// This code is running a CGSolve using a sparse matrix in ELL storage format
// We need three parallel operations: 
//   - axpby (vector add z = a*x + b*y)
//   - dot (dot product r = x*y)
//   - spmv (spare matrix - vector product y = A*x)

// EXERCISE: Use Kokkos Views here instead of pointers
void axpby(int64_t N, double* z, double a, double* x, double b, double* y) {
  // EXERCISE: make this a parallel loop
  for(int i=0; i<N; i++)
    z[i] = a*x[i] + b*y[i];
}

// EXERCISE: Use Kokkos Views here instead of pointers
double dot(int64_t N, double* x, double* y) {
  double result = 0;
  // EXERCISE: make this a parallel loop
  for(int i=0; i<N; i++)
    result += x[i] * y[i];
  return result;
};

// EXERCISE: Use Kokkos Views here instead of pointers
// col_idx and values should be 2D views for ELL Matrix format
// Do you see the 2D index calculation explicitly in this code?
void spmv(int64_t num_rows, 
          int* row_length, int64_t* col_idx, double* values,
          double* y, double* x) {
  int chunk_size = 16;
  int64_t num_chunks = (num_rows+chunk_size-1)/chunk_size;
  // EXERCISE-HP: what team_size and vector length would be good here depending on the backend?
  // EXERCISE-HP: use a TeamPolicy parallel loop here
  for(int64_t chunk = 0; chunk < num_chunks; chunk++) {
    int64_t start_row = chunk*chunk_size;
    int64_t end_row = start_row+chunk_size<num_rows?start_row+chunk_size:num_rows;
    // EXERCISE-HP: use a TeamThreadRange nested parallel loop here
    for(int64_t row = start_row; row<end_row; row++) {
      const int64_t length = row_length[row];

      double y_row = 0.;
      // EXERCISE-HP: use a ThreadVectorRange nested parallel loop here
      for(int64_t i=0; i<length; i++) {
        // EXERCISE: fix the indexing for Kokkos 2D Views
        int64_t idx = i * num_rows + row;
        y_row += values[idx] * x[col_idx[idx]];
      }
      y[row] = y_row;
    }
  }
}


// Runs a cgsolve cycle for num_iters iterations trying to solve b = A*y
void benchmark_cgsolve(int num_iters, double* x_in, ELLMatrix A, double* b_in) {
  double tolerance = 0;
  double normr     = 0;
  double rtrans    = 0;
  double oldrtrans = 0;

  int64_t print_freq = num_iters / 10;
  if (print_freq > 50) print_freq = 50;
  if (print_freq < 1) print_freq = 1;
  
  int64_t num_rows = A.num_rows();
  int64_t max_nnz_per_row = A.max_nnz_per_row();
  int* row_lengths_in = A.row_length;
  int64_t* col_idx_in = A.col_idx;
  double* values_in = A.values;

  // EXERCISE: create Kokkos::Views here for
  // x, b, row_lengths, col_idx, and values instead of copying pointers
  // Note: make values and col_idx 2D Views making the ELL format explicit!
  // EXERCISE: EXTRA: specify the layout explicitly or leave it off for col_idx and values, 
  // what is the performance impact?
  double* x = x_in;
  double* b = b_in;
  int* row_lengths = row_lengths_in;
  int64_t* col_idx = col_idx_in;
  double* values = values_in;

  // EXERCISE-HP: create HostMirror Views for x,b,row_lengths,cold-idx and values

  // EXERCISE: copy (even serially) the data from the fields in A, x_in and b_in to the
  // Kokkos Views.
  // Check spmv kernel to see correct index calculation for col_idx and values
  // For time reasons here is the solutions: make sure your naming and dimensions match!
  // EXERCISE: EXTRA: can you make this run in parallel on the host using the Kokkos::DefaultHostExecutionSpace?
  /*
  for(int row=0; row<num_rows; row++) {
    x_h(row) = x_in[row];
    b_h(row) = b_in[row];
    row_lengths_h(row) = row_lengths_in[row];
    for(int j=0; j<max_nnz_per_row; j++) {
      col_idx_h(row,j) = col_idx_in[row + num_rows * j];
      values_h(row,j) = values_in[row + num_rows * j];
    }
  }
  */

  // EXERCISE-HP: deep_copy x,b,row_lengths,cold-idx and values to the device views
  
  // EXERCISE: make all these here device views
  double* r = new double[num_rows];
  double* p = new double[num_rows];
  double* Ap = new double[num_rows];

  Kokkos::Timer timer;
  double one  = 1.0;
  double zero = 0.0;
  axpby(num_rows, p, one, x, zero, x);

  spmv(num_rows, row_lengths,col_idx,values,Ap,p);
  axpby(num_rows, r, one, b, -one, Ap);

  rtrans = dot(num_rows, r, r);

  normr = std::sqrt(rtrans);

  std::cout << "Initial Residual = " << normr << std::endl;

  double brkdown_tol = std::numeric_limits<double>::epsilon();

  for (int64_t k = 1; k <= num_iters && normr > tolerance; ++k) {
    if (k == 1) {
      axpby(num_rows, p, one, r, zero, r);
    } else {
      oldrtrans   = rtrans;
      rtrans      = dot(num_rows, r, r);
      double beta = rtrans / oldrtrans;
      axpby(num_rows, p, one, r, beta, p);
    }

    normr = std::sqrt(rtrans);

    if (k % print_freq == 0 || k == num_iters) {
      std::cout << "Iteration = " << k << "   Residual = " << normr
                << std::endl;
    }

    double alpha    = 0;
    double p_ap_dot = 0;

    spmv(num_rows,row_lengths,col_idx,values,Ap,p);

    p_ap_dot = dot(num_rows, Ap, p);

    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                  << std::endl;
      } else
        brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans / p_ap_dot;

    axpby(num_rows, x, one, x, alpha, p);
    axpby(num_rows, r, one, r, -alpha, Ap);
  }
  double time = timer.seconds();
  double gf_spmv = A.nnz() * 2;
  double gf_dot = num_rows * 2;
  double gf_axpby = num_rows * 4;
  double gflop = gf_axpby * 2 + gf_spmv + gf_dot +
                 num_iters * ( gf_axpby * 3 + gf_spmv + gf_dot * 2 );
  printf("DONE\n\nIterations: %i Residual: %e GFlop/s: %lf\n",num_iters, normr,
          1.e-9 * gflop / time);

  // EXERCISE: are the deletes still necessary?
  delete[] r;
  delete[] p;
  delete[] Ap;
}

int main(int argc, char* argv[]) {
  if(argc>1) {
    if((strcmp(argv[1],"-h")==0) ||
      (strcmp(argv[1],"--help")==0)) {
      printf("Kokkos CG-Solve Exercise\n");
      printf("  Args: N I [optional kokkos runtime arguments]\n");
      printf("  N: linear dimension of matrix, (N+1)^3 rows will be created.\n");
      printf("  I: number of cg-solve iterations to be run.\n");
      exit(0);
    }
  }
  // EXERCISE: initialize Kokkos
  int nx = argc>1?atoi(argv[1]):100;
  int num_iters = argc>2?atoi(argv[2]):200;

  printf("Setting up the problem for %li rows.\n",int64_t(nx+1)*int64_t(nx+1)*int64_t(nx+1));
  // This generates a CRS storage format matrix from the miniFE Mantevo app
  // The matrix represents a heat conduction problem in 3D
  CrsMatrix A = generate_miniFE_CrsMatrix(nx);
  // Create an ELL Storage format matrix from the CRSMatrix
  ELLMatrix B(A);

  // Create the x and y vector
  double* b = generate_miniFE_vector(nx); 
  double* x = new double[A.num_cols()];
  for(int64_t i=0; i<A.num_cols(); i++) x[i] = 0.0;

  benchmark_cgsolve(num_iters, x, B, b);

  delete[] b;
  delete[] x;
  A.free_data();
  B.free_data();
  // EXERCISE: finalize Kokkos
}
