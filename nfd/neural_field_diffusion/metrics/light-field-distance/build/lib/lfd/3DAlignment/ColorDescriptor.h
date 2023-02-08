#include <inttypes.h>

typedef int64_t __int64;

void ColorHistogram(double *HistogramValue, unsigned char *yuvBuff, int width,
                    int height);
void CompactColor(unsigned char *CompactValue, double *HistogramValue);
void ExtractCCD(unsigned char *YuvBuff, uint64_t *CCD, unsigned char *SrcBuff);
double ColorDistance(uint64_t *dest, uint64_t *src);
