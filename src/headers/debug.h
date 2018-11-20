// ==============================================================================
// This file is part of THOR.
//
//     THOR is free software : you can redistribute it and / or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     THOR is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//     GNU General Public License for more details.
//
//     You find a copy of the GNU General Public License in the main
//     THOR directory under <license.txt>.If not, see
//     <http://www.gnu.org/licenses/>.
// ==============================================================================
//
//
//
// Description: Defines debug parameters and enable helper functions
//
// Method: -
//
// Known limitations: None
//
//
// Known issues: None
//
//
// If you use this code please cite the following reference:
//
//       [1] Mendonca, J.M., Grimm, S.L., Grosheintz, L., & Heng, K., ApJ, 829, 115, 2016
//
// Current Code Owner: Joao Mendonca, EEG. joao.mendonca@csh.unibe.ch
//
// History:
// Version Date       Comment
// ======= ====       =======
//
// 1.0     16/08/2017 Released version  (JM)
//
////////////////////////////////////////////////////////////////////////

// benchmarking
// if defined run benchmark functions?
//#define BENCHMARKING

// ***************************************
// * binary comparison
// compare benchmark point to references
//#define BENCH_POINT_COMPARE
// write reference benchmark point
//#define BENCH_POINT_WRITE
// print out more debug info, by default, only print out failures
//#define BENCH_PRINT_DEBUG
// print out comparisaon statistics
//#define BENCH_COMPARE_PRINT_STATISTICS
// use an epsilon value for fuzzy compare on relative value
// #define BENCH_COMPARE_USE_EPSILON
// #define BENCH_COMPARE_EPSILON_VALUE 1e-7
// ***************************************
// * check for NaNs
// #define BENCH_NAN_CHECK
// #define BENCH_CHECK_LAST_CUDA_ERROR

// path to benchmark result directory
#define BENCHMARK_DUMP_REF_PATH "results/ref/"
#define BENCHMARK_DUMP_BASENAME "bindata_"


//#define GLOBAL_CONSERVATION_ATOMICADD
#define GLOBAL_CONSERVATION_REDUCTIONADD
//#define GLOBAL_CONSERVATION_CPUADD
