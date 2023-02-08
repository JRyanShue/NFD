#include "Morphology.h"
#include <stdlib.h>

void BINARYDilation3x3(BYTE *r, POINT ImageSize) {
  //			1  1  1
  //			1 <1> 1
  //			1  1  1
  POINT MaskCoor[9] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0},
                       {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  BINARYDilation(r, ImageSize, 9, MaskCoor);
}

void BINARYDilation5x5(BYTE *r, POINT ImageSize) {
  //			   1  1  1
  //			1  1  1  1  1
  //			1  1 <1> 1  1
  //			1  1  1  1  1
  //			   1  1  1
  POINT MaskCoor[21] = {{-2, -1}, {-2, 0}, {-2, 1}, {-1, -2}, {-1, -1}, {-1, 0},
                        {-1, 1},  {-1, 2}, {0, -2}, {0, -1},  {0, 0},   {0, 1},
                        {0, 2},   {1, -2}, {1, -1}, {1, 0},   {1, 1},   {1, 2},
                        {2, -1},  {2, 0},  {2, 1}};
  BINARYDilation(r, ImageSize, 21, MaskCoor);
}

void BINARYDilation(BYTE *r, POINT ImageSize, int MaskNum, POINT *MaskCoor) {
  BYTE *backup, *tmp;
  int TotalSize = ImageSize.x * ImageSize.y;
  int i, j, k, l;
  int tmpy;

  backup = (BYTE *)malloc(TotalSize * sizeof(BYTE));
  tmp = (BYTE *)malloc(TotalSize * sizeof(BYTE));

  memcpy(backup, r, TotalSize);

  for (i = 0; i < TotalSize; i++)
    *(r + i) = 0;

  for (i = 0; i < MaskNum; i++) {
    tmpy = (MaskCoor + i)->y * ImageSize.x;
    for (j = 0, l = 0; j < TotalSize; j += ImageSize.x, l++)
      for (k = 0; k < ImageSize.x; k++)
        if (k - (MaskCoor + i)->x >= 0 && k - (MaskCoor + i)->x < ImageSize.x &&
            l - (MaskCoor + i)->y >= 0 && l - (MaskCoor + i)->y < ImageSize.y &&
            *(backup + j - tmpy + k - (MaskCoor + i)->x) == 255)
          *(tmp + j + k) = 255;
        else
          *(tmp + j + k) = 0;
    for (j = 0; j < TotalSize; j++)
      *(r + j) = *(r + j) | *(tmp + j);
  }

  free(backup);
  free(tmp);
}

void BINARYErosion3x3(BYTE *r, POINT ImageSize) {
  //			1  1  1
  //			1 <1> 1
  //			1  1  1
  POINT MaskCoor[9] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0},
                       {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  BINARYErosion(r, ImageSize, 9, MaskCoor);
}

void BINARYErosion5x5(BYTE *r, POINT ImageSize) {
  //			   1  1  1
  //			1  1  1  1  1
  //			1  1 <1> 1  1
  //			1  1  1  1  1
  //			   1  1  1
  POINT MaskCoor[21] = {{-2, -1}, {-2, 0}, {-2, 1}, {-1, -2}, {-1, -1}, {-1, 0},
                        {-1, 1},  {-1, 2}, {0, -2}, {0, -1},  {0, 0},   {0, 1},
                        {0, 2},   {1, -2}, {1, -1}, {1, 0},   {1, 1},   {1, 2},
                        {2, -1},  {2, 0},  {2, 1}};
  BINARYErosion(r, ImageSize, 21, MaskCoor);
}

void BINARYErosion(BYTE *r, POINT ImageSize, int MaskNum, POINT *MaskCoor) {
  BYTE *backup;
  int TotalSize = ImageSize.x * ImageSize.y;
  int i, j, k;
  int tmpy;

  backup = (BYTE *)malloc(TotalSize * sizeof(BYTE));

  memcpy(backup, r, TotalSize);

  for (i = 0; i < MaskNum; i++) {
    tmpy = (MaskCoor + i)->y * ImageSize.x;
    // don't check boundary to speed-up
    for (j = 2 * ImageSize.x; j < TotalSize - 2 * ImageSize.x; j += ImageSize.x)
      for (k = 2; k < ImageSize.x - 2; k++)
        if (*(backup + j + tmpy + k + (MaskCoor + i)->x) < 255)
          *(r + j + k) = 0;
  }

  free(backup);
}

void BINARYOpening3x3(BYTE *r, POINT ImageSize) {
  BINARYErosion3x3(r, ImageSize);
  BINARYDilation3x3(r, ImageSize);
}

void BINARYOpening5x5(BYTE *r, POINT ImageSize) {
  BINARYErosion5x5(r, ImageSize);
  BINARYDilation5x5(r, ImageSize);
}

void BINARYClosing3x3(BYTE *r, POINT ImageSize) {
  BINARYDilation3x3(r, ImageSize);
  BINARYErosion3x3(r, ImageSize);
}

void BINARYClosing5x5(BYTE *r, POINT ImageSize) {
  BINARYDilation5x5(r, ImageSize);
  BINARYErosion5x5(r, ImageSize);
}

void BINARYHit_and_Miss_TopRightCorner(BYTE *r, POINT ImageSize) {
  // find right-top corner
  //			   0  0
  //			1 <1> 0
  //			   1
  POINT HitMask[3] = {{-1, 0}, {0, 0}, {0, 1}};
  // HitMask :
  //			1 <1>
  //			   1
  POINT MissMask[3] = {{0, -1}, {1, -1}, {1, 0}};
  // MissMask :
  //		    0  0
  //		   < > 0
  BINARYHit_and_Miss(r, ImageSize, HitMask, 3, MissMask, 3);
}

void BINARYHit_and_Miss_BottomLeftCorner(BYTE *r, POINT ImageSize) {
  // find right-top corner
  //			   1
  //			0 <1> 1
  //			0  0
  POINT HitMask[3] = {{0, -1}, {0, 0}, {1, 0}};
  // HitMask :
  //			1
  //		   <1> 1
  POINT MissMask[3] = {{-1, 0}, {-1, 1}, {0, 1}};
  // MissMask :
  //		    0 < >
  //		    0  0
  BINARYHit_and_Miss(r, ImageSize, HitMask, 3, MissMask, 3);
}

void BINARYHit_and_Miss(BYTE *r, POINT ImageSize, POINT *HitMask,
                        int HitMaskNum, POINT *MissMask, int MissMaskNum) {
  BYTE *hit, *miss;
  int TotalSize = ImageSize.x * ImageSize.y;
  int i;

  hit = (BYTE *)malloc(TotalSize * sizeof(BYTE));
  miss = (BYTE *)malloc(TotalSize * sizeof(BYTE));

  memcpy(hit, r, TotalSize);
  memcpy(miss, r, TotalSize);

  // do 0->1, 1->0
  for (i = 0; i < TotalSize; i++)
    *(miss + i) = *(miss + i) ^ 0x01;

  BINARYErosion(hit, ImageSize, HitMaskNum, HitMask);
  BINARYErosion(miss, ImageSize, MissMaskNum, MissMask);

  for (i = 0; i < TotalSize; i++)
    *(r + i) = *(hit + i) & *(miss + i);

  free(hit);
  free(miss);
}

void GRAYSCALEDilation(BYTE *r, POINT ImageSize, int MaskNum, POINT *MaskCoor,
                       int *MaskValue) {
  BYTE *backup;
  int TotalSize = ImageSize.x * ImageSize.y;
  int i, j, k, l;
  int max, tmp;

  backup = (BYTE *)malloc(TotalSize * sizeof(BYTE));
  memcpy(backup, r, TotalSize);

  for (i = 0; i < MaskNum; i++) {
    (MaskCoor + i)->x = -(MaskCoor + i)->x;
    (MaskCoor + i)->y = -(MaskCoor + i)->y;
  }

  for (j = 0, l = 0; j < TotalSize; j += ImageSize.x, l++)
    for (k = 0; k < ImageSize.x; k++) {
      max = 0;
      for (i = 0; i < MaskNum && max != 255; i++) {
        if (k + (MaskCoor + i)->x >= 0 && k + (MaskCoor + i)->x < ImageSize.x &&
            l + (MaskCoor + i)->y >= 0 && l + (MaskCoor + i)->y < ImageSize.y)
          tmp = *(MaskValue + i) +
                *(backup + (l + (MaskCoor + i)->y) * ImageSize.x + k +
                  (MaskCoor + i)->x);
        else
          continue;
        if (tmp > 255)
          tmp = 255;
        if (tmp > max)
          max = tmp;
      }
      *(r + j + k) = (BYTE)max;
    }

  free(backup);
}

void GRAYSCALEDilation3x3(BYTE *r, POINT ImageSize) {
  //			1  1  1
  //			1 <1> 1
  //			1  1  1
  POINT MaskCoor[9] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0},
                       {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  int MaskValue[9] = {36, 36, 36, 36, 36, 36, 36, 36, 36};

  GRAYSCALEDilation(r, ImageSize, 9, MaskCoor, MaskValue);
}

void GRAYSCALEDilation5x5(BYTE *r, POINT ImageSize) {
  //			   1  1  1
  //			1  1  1  1  1
  //			1  1 <1> 1  1
  //			1  1  1  1  1
  //			   1  1  1
  POINT MaskCoor[21] = {{-2, -1}, {-2, 0}, {-2, 1}, {-1, -2}, {-1, -1}, {-1, 0},
                        {-1, 1},  {-1, 2}, {0, -2}, {0, -1},  {0, 0},   {0, 1},
                        {0, 2},   {1, -2}, {1, -1}, {1, 0},   {1, 1},   {1, 2},
                        {2, -1},  {2, 0},  {2, 1}};
  int MaskValue[21] = {36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                       36, 36, 36, 36, 36, 36, 36, 36, 36, 36};

  GRAYSCALEDilation(r, ImageSize, 21, MaskCoor, MaskValue);
}

void GRAYSCALEErosion(BYTE *r, POINT ImageSize, int MaskNum, POINT *MaskCoor,
                      int *MaskValue) {
  BYTE *backup;
  int TotalSize = ImageSize.x * ImageSize.y;
  int i, j, k, l;
  int min, tmp;

  backup = (BYTE *)malloc(TotalSize * sizeof(BYTE));
  memcpy(backup, r, TotalSize);

  for (j = 0, l = 0; j < TotalSize; j += ImageSize.x, l++)
    for (k = 0; k < ImageSize.x; k++) {
      min = 255;
      for (i = 0; i < MaskNum && min != 0; i++) {
        if (k + (MaskCoor + i)->x >= 0 && k + (MaskCoor + i)->x < ImageSize.x &&
            l + (MaskCoor + i)->y >= 0 && l + (MaskCoor + i)->y < ImageSize.y)
          tmp = *(backup + (l + (MaskCoor + i)->y) * ImageSize.x + k +
                  (MaskCoor + i)->x) -
                *(MaskValue + i);
        else
          continue;
        if (tmp < 0)
          tmp = 0;
        if (tmp < min)
          min = tmp;
      }
      *(r + j + k) = (BYTE)min;
    }

  free(backup);
}

void GRAYSCALEErosion3x3(BYTE *r, POINT ImageSize) {
  //			1  1  1
  //			1 <1> 1
  //			1  1  1
  POINT MaskCoor[9] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0},
                       {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  int MaskValue[9] = {36, 36, 36, 36, 36, 36, 36, 36, 36};

  GRAYSCALEErosion(r, ImageSize, 9, MaskCoor, MaskValue);
}

void GRAYSCALEErosion5x5(BYTE *r, POINT ImageSize) {
  //			   1  1  1
  //			1  1  1  1  1
  //			1  1 <1> 1  1
  //			1  1  1  1  1
  //			   1  1  1
  POINT MaskCoor[21] = {{-2, -1}, {-2, 0}, {-2, 1}, {-1, -2}, {-1, -1}, {-1, 0},
                        {-1, 1},  {-1, 2}, {0, -2}, {0, -1},  {0, 0},   {0, 1},
                        {0, 2},   {1, -2}, {1, -1}, {1, 0},   {1, 1},   {1, 2},
                        {2, -1},  {2, 0},  {2, 1}};
  int MaskValue[21] = {36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                       36, 36, 36, 36, 36, 36, 36, 36, 36, 36};

  GRAYSCALEErosion(r, ImageSize, 21, MaskCoor, MaskValue);
}

void GRAYSCALEOpening3x3(BYTE *r, POINT ImageSize) {
  GRAYSCALEErosion3x3(r, ImageSize);
  GRAYSCALEDilation3x3(r, ImageSize);
}

void GRAYSCALEOpening5x5(BYTE *r, POINT ImageSize) {
  GRAYSCALEErosion5x5(r, ImageSize);
  GRAYSCALEDilation5x5(r, ImageSize);
}

void GRAYSCALEClosing3x3(BYTE *r, POINT ImageSize) {
  GRAYSCALEDilation3x3(r, ImageSize);
  GRAYSCALEErosion3x3(r, ImageSize);
}

void GRAYSCALEClosing5x5(BYTE *r, POINT ImageSize) {
  GRAYSCALEDilation5x5(r, ImageSize);
  GRAYSCALEErosion5x5(r, ImageSize);
}

unsigned char BoundingBox(unsigned char *mY, POINT Size) {
  int i, ii, j;
  int left, right, top, down;

  left = top = 1000000;
  right = down = -1;

  for (i = 0, ii = 0; i < Size.y; i++, ii += Size.x)
    for (j = 0; j < Size.x; j++)
      if (mY[ii + j] < 255) {
        if (i < top)
          top = i;
        if (i > down)
          down = i;
        if (j < left)
          left = j;
        if (j > right)
          right = j;
      }

  if (right >= 0) {
    for (i = top, ii = top * Size.x; i <= down; i++, ii += Size.x)
      for (j = left; j <= right; j++)
        mY[ii + j] = 0;

    return 1; // ok
  } else
    return -1; // no pixel
}
