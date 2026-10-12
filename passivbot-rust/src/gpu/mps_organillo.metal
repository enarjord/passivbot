// Optional external entry eligibility. Empty macros preserve the ordinary ABI.
#ifdef PASSIVBOT_ORGANILLO
#ifdef __CUDACC__
using uchar = unsigned char;
#endif
#define ORGANILLO_ARGS , constant uchar* organillo_values, constant int* organillo_rows
#define ORGANILLO_PASS , organillo_values, organillo_rows
#define ORGANILLO_ALLOWED(k, c, coins) (organillo_values[organillo_rows[k] * (coins) + (c)] != 0)
#else
#define ORGANILLO_ARGS
#define ORGANILLO_PASS
#define ORGANILLO_ALLOWED(k, c, coins) true
#endif
