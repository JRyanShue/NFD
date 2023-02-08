#include "TypeConverts.h"
void BINARYDilation(BYTE *r, POINT ImageSize, int MaskNum, POINT *MaskCoor);
void BINARYDilation3x3(BYTE *r, POINT ImageSize);
void BINARYDilation5x5(BYTE *r, POINT ImageSize);
void BINARYErosion(BYTE *r, POINT ImageSize, int MaskNum, POINT *MaskCoor);
void BINARYErosion3x3(BYTE *r, POINT ImageSize);
void BINARYErosion5x5(BYTE *r, POINT ImageSize);
void BINARYOpening3x3(BYTE *r, POINT ImageSize);
void BINARYOpening5x5(BYTE *r, POINT ImageSize);
void BINARYClosing3x3(BYTE *r, POINT ImageSize);
void BINARYClosing5x5(BYTE *r, POINT ImageSize);
void BINARYHit_and_Miss(BYTE *r, POINT ImageSize, POINT *HitMask,
                        int HitMaskNum, POINT *MissMask, int MissMaskNum);
void BINARYHit_and_Miss_TopRightCorner(BYTE *r, POINT ImageSize);
void BINARYHit_and_Miss_BottomLeftCorner(BYTE *r, POINT ImageSize);
void GRAYSCALEDilation(BYTE *r, POINT ImageSize, int MaskNum, POINT *MaskCoor,
                       int *MaskValue);
void GRAYSCALEDilation3x3(BYTE *r, POINT ImageSize);
void GRAYSCALEDilation5x5(BYTE *r, POINT ImageSize);
void GRAYSCALEErosion(BYTE *r, POINT ImageSize, int MaskNum, POINT *MaskCoor,
                      int *MaskValue);
void GRAYSCALEErosion3x3(BYTE *r, POINT ImageSize);
void GRAYSCALEErosion5x5(BYTE *r, POINT ImageSize);
void GRAYSCALEOpening3x3(BYTE *r, POINT ImageSize);
void GRAYSCALEOpening5x5(BYTE *r, POINT ImageSize);
void GRAYSCALEClosing3x3(BYTE *r, POINT ImageSize);
void GRAYSCALEClosing5x5(BYTE *r, POINT ImageSize);
unsigned char BoundingBox(unsigned char *mY, POINT Size);
