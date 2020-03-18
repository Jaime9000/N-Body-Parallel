#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "upc_relaxed.h"
#include <upc_collective.h>

#define N 10000
#define G 6.67e-11
#define TIMESTEP 0.25
#define NSTEPS 10



/*-------Structures--Prototypes--&-Declerations------*/

/*
 * body data structure
 */
struct body_s {
  double x;
  double y;
  double z;
  double dx;
  double dy;
  double dz;
  double mass;
};
typedef struct body_s body_t;


/*
 * function prototypes
 */
void init(void);
double dist(double dx, double dy, double dz);
void upc_all_gather_all(shared void*dst,shared const void*src, 
    size_t nbytes, upc_flag_t sync_mode);

shared[THREADS] body_t bodies[N];  // array of N-bodies at timestep t
shared[THREADS] body_t next[N];    // array of N-bodies at timestep t+1

shared int globalarray[THREADS];

/*------Formality Functions...(tools for output-------*/

/*
 * get_wctime - returns wall clock time as double
 *   @return double representation of wall clock time
 */
double get_wctime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

/**
 *  eprintf - error printing wrapper (only prints once)
 *    @param format printf-like format string
 *    @param ... arguments to printf format string
 *    @return number of bytes written to stderr
 */
int eprintf(const char *format, ...) {
  va_list ap;
  int ret;

  if (MYTHREAD == 0) {
    va_start(ap, format);
    ret = vfprintf(stdout, format, ap);
    va_end(ap);
    return ret;
  }
  else
    return 0;
}



/*------------------Actual Math Functions-------------------------------*/


/**
 * init - give the planets initial values for position, velocity, mass
 */
void init(void) {

upc_forall (int i=0; i<N; i++,i) {
    bodies[i].x = 100.0 * (i + 0.1);
    bodies[i].y = 200.0 * (i + 0.1);
    bodies[i].z = 300.0 * (i + 0.1);
    bodies[i].dx = i + 400.0;
    bodies[i].dy = i + 500.0;
    bodies[i].dz = i + 600.0;
    bodies[i].mass = 10e6 * (i + 100.2);
  }
}

/**
 * dist - determine the distance between two bodies
 *    @param dx - distance in the x dimension
 *    @param dy - distance in the y dimension
 *    @param dz - distance in the z dimension
 *    @return distance
 */
double dist(double dx, double dy, double dz) {
  return sqrt((dx*dx) + (dy*dy) + (dz*dz));;
}

/*
 * Declare PRIVATE/SHARED pointer(s) for head access of global array
 * Creates private pointer me & private pointer nextme for each thread
 * All copies of the two pointers function the same
 * And have explicit affinity(association) with their specific thread
 */
 //body_t *me, body_t *nextme;

/**
 * computeforce - compute the superposed forces on one body
 *   @param me     - the body to compute forces on at time t
 *   @param nextme - the body at time t+1
 */
void computeforce(body_t *me, body_t *nextme) {
  double d, f;        // distance, force
  double dx, dy, dz;  // position deltas
  double fx, fy, fz;  // force components
  double ax, ay, az;  // acceleration components

  fx = fy = fz = 0.0;
	
	upc_barrier(); //everything together 

  upc_forall (int i=0; i<N; i++,i) {

    // compute the distances in each dimension
    dx = me->x - bodies[i].x;
    dy = me->y - bodies[i].y;
    dz = me->z - bodies[i].z;

    // compute the distance magnitude
    d = dist(dx, dy, dz);

    // skip over ourselves (d==0)
    if (d != 0) {

      // F = G m1 m2 / r^2
      f = (G * me->mass * bodies[i].mass) / (d * d);

      // compute force components in each dimension
      fx += (f * dx) / d;
      fy += (f * dy) / d;
      fz += (f * dz) / d;

      // acc = force / mass (F=ma)
      ax = fx / me->mass;
      ay = fy / me->mass;
      az = fz / me->mass;

      // update the body velocity at time t+1
      nextme->dx = me->dx + (TIMESTEP * ax);
      nextme->dy = me->dy + (TIMESTEP * ay);
      nextme->dz = me->dz + (TIMESTEP * az);

      // update the body position at t+1
      nextme->x = me->x + (TIMESTEP * me->dx);
      nextme->y = me->y + (TIMESTEP * me->dy);
      nextme->z = me->z + (TIMESTEP * me->dz);

      // copy over the mass
      nextme->mass = me-> mass;
    }
  }
}



/**
 * main
 */
int main(int argc, char **argv) {
  double start;
  
  //set initial maths
  init();


  eprintf("beginning N-body simulation of %d bodies with %d processes.\n", N, THREADS);

  setbuf(stdout, NULL);

  start = get_wctime();

  // for each timestep in the simulation
  upc_forall (int ts=0; ts<NSTEPS; ts++; ts){
    /*for each step, broadcast the ((n/P)-1)^th AND ((n/P))th thread
     * Gather((n/P)-1)--> now 
     * Gather((n/P)))--> 
     */

    // for every body in the universe
    upc_forall(int i=0; i<N; i++,i){
      computeforce(&bodies[i], &next[i]);
    }



    

  upc_forall (int i=0; i<THREADS; i++; &(globalarray[i])) {
    printf("thread %d setting element globalarray[%d] : owner is %d\n",
        MYTHREAD, i, (int)upc_threadof(&(globalarray[i])));

    globalarray[i] = i;
  }

  upc_barrier;
  sleep(1);

 /* for (int i=0; i<THREADS; i++) {
    if (MYTHREAD == i) {
      printf("hello from %d of %d - globalarray[%d] is %d\n",
          MYTHREAD, THREADS, i, globalarray[i]);
    }
    upc_barrier;
  }*/
	
	upc_barrier;

  for (int i=0; i<THREADS; i++) {
    if (i == MYTHREAD) {
      printf("hello from %d\n", MYTHREAD);
    }
    upc_barrier;
  }
	
  upc_barrier;

  eprintf("execution time: %7.4f ms\n", (get_wctime()-start)*1000);
  return 0;




}

