# include <stdio.h>
# include <stdlib.h>
# include <stdbool.h>
# include <assert.h>
# include <math.h>
# include <time.h>

struct simplex_t {
    int dim;
    double *pdf;
    double *cdf;
};

double rand_u () {
    return rand() / (RAND_MAX + 1.0);
}

double rand_exp (double lambda) {
    return -log(1 - rand_u()) / lambda;
}

int rand_int (int n, int m) {
    // n <= ret < m

    return n + (int)(rand_u() * (m - n));
}

struct simplex_t *init_simplex (int dim) {
    struct simplex_t *simplex = malloc(sizeof(struct simplex_t));
    double sum = 0.;
    int i;

    simplex->dim = dim;
    simplex->pdf = malloc(sizeof(double) * (dim + 1));
    simplex->cdf = malloc(sizeof(double) * (dim + 1));

    simplex->pdf[0] = 0.;
    simplex->cdf[0] = 0.;

    return simplex;
}

void normalize_simplex (struct simplex_t *simplex) {
    int i;
    double sum = 0;

    for (i = 1; i <= simplex->dim; ++i)
        sum += simplex->pdf[i];
    for (i = 1; i <= simplex->dim; ++i) {
        simplex->pdf[i] /= sum;
        simplex->cdf[i]  = simplex->cdf[i - 1] + simplex->pdf[i];
    }
}

void dest_simplex (struct simplex_t *simplex) {
    free(simplex->pdf);
    free(simplex->cdf);
    free(simplex);
}

struct simplex_t *uniform_simplex (int dim) {
    struct simplex_t *simplex = init_simplex(dim);
    int i;

    for (i = 1; i <= dim; ++i)
        simplex->pdf[i] = 1;

    normalize_simplex(simplex);

    return simplex;
}

struct simplex_t *sample_simplex (int dim) {
    struct simplex_t *simplex = init_simplex(dim);
    int i;

    for (i = 1; i <= dim; ++i)
        simplex->pdf[i] = rand_exp(1.);

    normalize_simplex(simplex);

    return simplex;
}

int guess (int lo, int hi, struct simplex_t *simplex) {
    // Best guess lo < x < hi

    double prob_all = simplex->cdf[hi - 1] - simplex->cdf[lo];
    double prob_0   = simplex->cdf[lo];
    double prob;
    int mid;

    assert(hi - lo > 1);

    // B-search
    while (true) {
        if (hi - lo == 2)
            return hi - 1;

        mid = (lo + hi) >> 1;
        // fprintf(stderr, "%d %d %d\n", lo, hi, mid);
        prob = (simplex->cdf[mid] - prob_0) * 2.;
        /* fprintf(stderr, "%d %d %lf %lf %lf %lf %lf\n",
                mid_d, mid_u, simplex->cdf[mid_d], simplex->cdf[mid_u],
                prob_d, prob_u, prob_all); */
        if (prob <= prob_all){
            lo = mid;
        }
        else{
            if (mid == lo + 1)
                return mid;
            hi = mid;
        }
    }
}

int mid_guess (int lo, int hi, struct simplex_t *simplex) {
    // mid guess lo + hi / 2

    return (lo + hi) >> 1;
}


int rand_guess (int lo, int hi, struct simplex_t *simplex) {
    // Random guess

    return rand_int(lo + 1, hi);
}

int game (int ans, struct simplex_t *simplex, 
        int (*guess_method)(int, int, struct simplex_t*)) {
    // lets play a game

    int len = 1;
    int lo  = 0;
    int hi  = simplex->dim + 1;
    int mid;

    while (true) {
        len += 1;
        mid  = guess_method(lo, hi, simplex);

        if (mid == ans)
            return len;
        else if (mid > ans)
            hi = mid;
        else
            lo = mid;
    }
}

double comp_avg_len (struct simplex_t *simplex, 
        int (*guess_method)(int, int, struct simplex_t*)) {
    double len = 0.;
    int i;

    for (i = 1; i <= simplex->dim; ++i)
        len += simplex->pdf[i] * game(i, simplex, guess_method);

    return len;
}

double comp_entropy (struct simplex_t *simplex) {
    double entropy = 0.;
    int i;

    for (i = 1; i <= simplex->dim; ++i)
        entropy += -simplex->pdf[i] * log2(simplex->pdf[i]);

    return entropy;
}

void simulate(int n, int repeat, 
        double *avg_len, double *avg_len_m, double *avg_len_r,
        double *entropy) {
    int i;
    struct simplex_t *simplex;
    
    #pragma omp parallel for num_threads (32)
    for (i = 0; i < repeat; ++i) {
        struct simplex_t *simplex = sample_simplex(n);
        avg_len[i]   = comp_avg_len(simplex, guess);
        avg_len_m[i] = comp_avg_len(simplex, mid_guess);
        avg_len_r[i] = comp_avg_len(simplex, rand_guess);
        entropy[i]   = comp_entropy(simplex);
        dest_simplex(simplex);
    }
}

# define R   16
# define N   240
# define N0  2
# define Del 0.05
// N0 <= N0 + Del * i < N0 + Del * N

double entropy[R * N];
double avg_len[R * N];
double avg_len_m[R * N];
double avg_len_r[R * N];

int main(void) {
    int i;
    int n;

    srand((unsigned)time(NULL));

    for (i = 0; i < N; ++i) {
        n = (int)(pow(2., N0 + Del * i));
        fprintf(stderr, "%d\n", n);
        simulate(n, R, avg_len + i * R, avg_len_m + i * R, avg_len_r + i * R,
                entropy + i * R);
    }

    for (i = 0; i < R * N; ++i)
        printf("%lf, %lf, %lf, %lf\n", entropy[i],
                avg_len[i], avg_len_m[i], avg_len_r[i]);

    return 0;
}
