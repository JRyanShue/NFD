#include <float.h>
#include <limits.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds.h"

// ****************
// NNNNOOOOTTTTEEEE: the ModelNum should be the same as the number of all models
#define ModelNum 2

// #1
unsigned char q8_table[256][256];

unsigned char src_ArtCoeff[SRC_ANGLE][CAMNUM][ART_COEF];
unsigned char dest_ArtCoeff[SRC_ANGLE][CAMNUM][ART_COEF];

unsigned char align10[60][CAMNUM_2];

int cost[SRC_ANGLE][ANGLE][CAMNUM][CAMNUM];

// for Fourier Descriptor matching
unsigned char src_FdCoeff_q8[ANGLE][CAMNUM][FD_COEF];
unsigned char dest_FdCoeff_q8[ANGLE][CAMNUM][FD_COEF];

// for Circularity
unsigned char src_CirCoeff_q8[ANGLE][CAMNUM];
unsigned char dest_CirCoeff_q8[ANGLE][CAMNUM];

// for Eccentricity
unsigned char src_EccCoeff_q8[ANGLE][CAMNUM];
unsigned char dest_EccCoeff_q8[ANGLE][CAMNUM];

#define abs(a) ((a) > 0) ? (a) : (-(a))
#define RESAM 1000

//******************************************************************************************************
// match from 20 shapes, and match 36 coeff for each shape, and each coeff has 4
// bits
int MatchART_q8(unsigned char dest_ArtCoeff[SRC_ANGLE][CAMNUM][ART_COEF]) {
  int i, j, srcCam, destCam;
  register unsigned char m;
  int err, MinErr;
  int distance;
  static int cost_q8[ANGLE][ANGLE][CAMNUM][CAMNUM];

  // compare each coefficients pair from the two models first
  for (srcCam = 0; srcCam < ANGLE; srcCam++)
    for (destCam = 0; destCam < ANGLE; destCam++)
      for (i = 0; i < CAMNUM; i++)
        for (j = 0; j < CAMNUM; j++) {
          // un-loop to speed-up
          // un-loop to speed-up
          m = 0;
          distance = q8_table[dest_ArtCoeff[destCam][i][m]]
                             [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          // 11
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          // 21
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          // 31
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];
          distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                              [src_ArtCoeff[srcCam][j][m++]];

          cost_q8[srcCam][destCam][i][j] = distance;
        }

  // find minimum error of the two models from all camera pairs
  MinErr = INT_MAX;
  for (srcCam = 0; srcCam < SRC_ANGLE; srcCam++)  // each src angle
    for (destCam = 0; destCam < ANGLE; destCam++) // each dest angle
      for (j = 0; j < 60; j++)                    // each align
      {
        //					err = 0;
        //					for(m=0; m<CAMNUM; m++)
        //// each vertex 						err +=
        //cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];

        // un-loop to speed-up
        m = 0;
        err = cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];

        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];

        if (err < MinErr)
          MinErr = err;
      }

  return MinErr;
  //	return MinErr<<3;
}

int MatchFD_q8(unsigned char dest_FdCoeff_q8[ANGLE][CAMNUM][FD_COEF]) {
  int i, j, srcCam, destCam;
  register unsigned char m;
  int err, MinErr;
  int distance;
  static int cost_q8[ANGLE][ANGLE][CAMNUM][CAMNUM];

  // compare each coefficients pair from the two models first
  for (srcCam = 0; srcCam < ANGLE; srcCam++)
    for (destCam = 0; destCam < ANGLE; destCam++)
      for (i = 0; i < CAMNUM; i++)
        for (j = 0; j < CAMNUM; j++) {
          // un-loop to speed-up
          m = 0;
          distance = q8_table[dest_FdCoeff_q8[destCam][i][m]]
                             [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];
          distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                              [src_FdCoeff_q8[srcCam][j][m++]];

          cost_q8[srcCam][destCam][i][j] = distance;
        }

  // find minimum error of the two models from all camera pairs
  MinErr = INT_MAX;
  for (srcCam = 0; srcCam < SRC_ANGLE; srcCam++)  // each src angle
    for (destCam = 0; destCam < ANGLE; destCam++) // each dest angle
      for (j = 0; j < 60; j++)                    // each align
      {
        //					err = 0;
        //					for(m=0; m<CAMNUM; m++)
        //// each vertex 						err +=
        //cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];

        // un-loop to speed-up
        m = 0;
        err = cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];

        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];

        if (err < MinErr)
          MinErr = err;
      }

  return MinErr << 1;
}

// matching FightField descriptor ( ART + Fourier )
int MatchLF(unsigned char dest_ArtCoeff[SRC_ANGLE][CAMNUM][ART_COEF],
            unsigned char dest_FdCoeff_q8[ANGLE][CAMNUM][FD_COEF],
            unsigned char dest_CirCoeff_q8[ANGLE][CAMNUM],
            unsigned char dest_EccCoeff_q8[ANGLE][CAMNUM]) {
  int i, j, srcCam, destCam;
  register unsigned char m;
  int err, MinErr;
  int art_distance, fd_distance, cir_distance, ecc_distance;
  static int cost_q8[ANGLE][ANGLE][CAMNUM][CAMNUM];

  // compare each coefficients pair from the two models first
  for (srcCam = 0; srcCam < ANGLE; srcCam++)
    for (destCam = 0; destCam < ANGLE; destCam++)
      for (i = 0; i < CAMNUM; i++)
        for (j = 0; j < CAMNUM; j++) {
          // un-loop to speed-up
          // for ART (Zernike moment)
          m = 0;
          art_distance = q8_table[dest_ArtCoeff[destCam][i][m]]
                                 [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          // 11
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          // 21
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          // 31
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];
          art_distance += q8_table[dest_ArtCoeff[destCam][i][m]]
                                  [src_ArtCoeff[srcCam][j][m++]];

          // for Fourier Descriptor
          m = 0;
          fd_distance = q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance += q8_table[dest_FdCoeff_q8[destCam][i][m]]
                                 [src_FdCoeff_q8[srcCam][j][m++]];
          fd_distance <<= 1;

          // for Circularity
          cir_distance = q8_table[dest_CirCoeff_q8[destCam][i]]
                                 [src_CirCoeff_q8[srcCam][j]];
          cir_distance <<= 1;

          // for Eccentricity
          ecc_distance = q8_table[dest_EccCoeff_q8[destCam][i]]
                                 [src_EccCoeff_q8[srcCam][j]];
          // ecc_distance <<= 1;

          cost_q8[srcCam][destCam][i][j] =
              art_distance + fd_distance + cir_distance + ecc_distance;
        }

  // find minimum error of the two models from all camera pairs
  MinErr = INT_MAX;
  for (srcCam = 0; srcCam < SRC_ANGLE; srcCam++)  // each src angle
    for (destCam = 0; destCam < ANGLE; destCam++) // each dest angle
      for (j = 0; j < 60; j++)                    // each align
      {
        m = 0;
        err = cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];
        err += cost_q8[srcCam][destCam][align10[j][m]][align10[0][m++]];

        if (err < MinErr)
          MinErr = err;
      }

  return MinErr;
}

void printHelp() {
  printf("Usage: \n");
  printf("\t ./GroundTruth obj1 obj2\n");
  printf("obj1 - name of the first object to be compared\n");
  printf("obj2 - name of the second object to be compared\n");
}

void readDestObjCoeffs(char *objname) {
  char filename[200];
  FILE *fpt;
  sprintf(filename, "%s_q8_v1.8.art", objname);
  if ((fpt = fopen(filename, "rb")) == NULL) {
    printf("%s does not exist.\n", filename);
    exit(1);
  }
  fread(dest_ArtCoeff, ANGLE * CAMNUM * ART_COEF, sizeof(unsigned char), fpt);
  fclose(fpt);
  sprintf(filename, "%s_q8_v1.8.fd", objname);
  if ((fpt = fopen(filename, "rb")) == NULL) {
    printf("%s does not exist.\n", filename);
    exit(1);
  }
  fread(dest_FdCoeff_q8, sizeof(unsigned char), ANGLE * CAMNUM * FD_COEF, fpt);
  fclose(fpt);
  sprintf(filename, "%s_q8_v1.8.cir", objname);
  if ((fpt = fopen(filename, "rb")) == NULL) {
    printf("%s does not exist.\n", filename);
    exit(1);
  }
  fread(dest_CirCoeff_q8, sizeof(unsigned char), ANGLE * CAMNUM, fpt);
  fclose(fpt);
  sprintf(filename, "%s_q8_v1.8.ecc", objname);
  if ((fpt = fopen(filename, "rb")) == NULL) {
    printf("%s does not exist.\n", filename);
    exit(1);
  }
  fread(dest_EccCoeff_q8, sizeof(unsigned char), ANGLE * CAMNUM, fpt);
}

void readSrcObjCoeffs(char *objname) {
  char filename[200];
  FILE *fpt;
  sprintf(filename, "%s_q8_v1.8.art", objname);
  if ((fpt = fopen(filename, "rb")) == NULL) {
    printf("%s does not exist.\n", filename);
    return;
  }
  fread(src_ArtCoeff, SRC_ANGLE * CAMNUM * ART_COEF, sizeof(unsigned char),
        fpt);
  fclose(fpt);
  // FD
  sprintf(filename, "%s_q8_v1.8.fd", objname);
  if ((fpt = fopen(filename, "rb")) == NULL) {
    printf("%s does not exist.\n", filename);
    return;
  }
  fread(src_FdCoeff_q8, sizeof(unsigned char), ANGLE * CAMNUM * FD_COEF, fpt);
  fclose(fpt);
  // CIR
  sprintf(filename, "%s_q8_v1.8.cir", objname);
  if ((fpt = fopen(filename, "rb")) == NULL) {
    printf("%s does not exist.\n", filename);
    return;
  }
  fread(src_CirCoeff_q8, sizeof(unsigned char), ANGLE * CAMNUM, fpt);
  fclose(fpt);
}

void main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("ERROR: The program has only 2 arguments, received %d instead!\n",
           argc - 1);
    printHelp();
    exit(2);
  }
  FILE *fpt;

  fpt = fopen("q8_table", "rb");
  fread(q8_table, sizeof(unsigned char), 65536, fpt);
  fclose(fpt);

  // initialize: read camera pair
  fpt = fopen("align10.txt", "rb");
  fread(align10, sizeof(unsigned char), 60 * CAMNUM_2, fpt);
  fclose(fpt);

  // read feature of all models
  readDestObjCoeffs(argv[1]);
  readSrcObjCoeffs(argv[2]);
  // initialize

  int similarity = MatchLF(dest_ArtCoeff, dest_FdCoeff_q8, dest_CirCoeff_q8,
                           dest_EccCoeff_q8);

  printf("\nSIMILARITY: %d\n", similarity);
}
