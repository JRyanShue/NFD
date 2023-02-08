#include "glut.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/osmesa.h>
#include <inttypes.h>

#include <stdio.h>
#include <malloc.h>
#include <memory.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <limits.h>

#include "TypeConverts.h"
#include "ds.h"
#include "RWObj.h"
#include "Bitmap.h"
#include "TranslateScale.h"
#include "Rotate.h"
#include "RegionShape.h"
#include "RecovAffine.h"
#include "Refine.h"
#include "edge.h"
#include "convert.h"
#include "ColorDescriptor.h"
#include "Eccentricity.h"
#include "Circularity.h"
#include "FourierDescriptor.h"

#define abs(a) (a>0)?(a):-(a)

#define	QUANT8				256		// 2^8
#define FD_SCALE			2		// *2 first, and then quantization
#define CIR_SCALE			2.318181818		// the range of circularity is [0~110], so *2.318 to be [0~255]
#define ECC_SCALE			25.5			// the range of circularity is [0~10], so *25.5 to be [0~255]

unsigned char	CamMap[CAMNUM_2]={0,1,2,3,4,5,6,7,8,9,5,6,7,8,9,2,3,4,0,1};

char srcfn[100];
char destfn[100];
GLubyte buffer;

int			winw = WIDTH, winh = HEIGHT;

pVer		vertex=NULL;
pTri		triangle=NULL;
int			NumVer=0, NumTri=0;		// total number of vertex and triangle.

pVer		vertex1, vertex2;
pTri		triangle1, triangle2;
int			NumVer1, NumTri1, NumVer2, NumTri2;		// total number of vertex and triangle.

// translate and scale of model 1 and 2
Ver				Translate1, Translate2;
double			Scale1, Scale2;
char *fname;

void FindCenter(unsigned char *srcBuff, int width, int height, double *CenX, double *CenY)
{
	int					x, y, count;
	unsigned char		*pImage;
	int					maxX, minX, maxY, minY;
	int					MeanX, MeanY; 

	count = 0;
	pImage = srcBuff;


	// uee center of max and min to be center
	maxX = maxY = -1;
	minX = minY = INT_MAX;
	for (y=0 ; y<height ; y++)
	for (x=0 ; x<width; x++)
	{
		if( *pImage < 255 )
		{
			if( x > maxX ) maxX = x;
			if( x < minX ) minX = x;
			if( y > maxY ) maxY = y;
			if( y < minY ) minY = y;
		}
		pImage++;
	}

	if( maxX > 0 )
	{
		*CenX = (maxX+minX) / 2.0;
		*CenY = (maxY+minY) / 2.0;
	}
	else
		*CenX = *CenY = -1;		// nothing to be rendered

}

void display(void)
{
	int				i, j;
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();				
		for(i = 0; i<NumTri; i++)
		{
			glColor3f((GLfloat)triangle[i].r, (GLfloat)triangle[i].g, (GLfloat)triangle[i].b);
			glBegin(GL_POLYGON);
      for(j=0; j<triangle[i].NodeName; j++){
        glVertex3d(vertex[triangle[i].v[j]].coor[0], vertex[triangle[i].v[j]].coor[1], vertex[triangle[i].v[j]].coor[2]);
      }
			glEnd();
		}
	glPopMatrix();				
  glutSwapBuffers();
}

void RenderToMem(unsigned char *bmBits, unsigned char *bmColor, pVer CamVertex, pVer v, pTri t, int nv, int nt)
{
  glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(CamVertex->coor[0], CamVertex->coor[1], CamVertex->coor[2],
				0, 0, 0,
				0, 1, 0);
	vertex = v;
	triangle = t;
	NumVer = nv;
	NumTri = nt;
	display();

  glReadBuffer(GL_BACK);
	glReadPixels(0, 0, winw, winh, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, bmBits);

	if( bmColor ) {
		glReadPixels(0, 0, winw, winh, GL_RGB, GL_UNSIGNED_BYTE, bmColor);
  }
}

void renderingFun()
{
	unsigned char	*srcBuff[CAMNUM], *destBuff[CAMNUM], *EdgeBuff, *ColorBuff[CAMNUM], *YuvBuff;
	char			filename[400];
	pVer			CamVertex[ANGLE];
	pTri			CamTriangle[ANGLE];
	int				CamNumVer[ANGLE], CamNumTri[ANGLE];		// total number of vertex and triangle.
	FILE			*fpt, *fpt2, *fpt3, *fpt4, *fpt_art_q8, *fpt_art_q4, *fpt_fd_q8, *fpt_fd, *fpt_cir_q8, *fpt_ecc_q8;//, *fpt_ccd;
	int				i, j, k, srcCam, destCam, p, r, a, itmp;
	double			cost[ANGLE][ANGLE][CAMNUM_2][CAMNUM_2];
	double			**matrix;
	static int		UseCam = 2;
	double			Err;
	// for region shape descriptor
	double			src_ArtCoeff[ANGLE][CAMNUM][ART_ANGULAR][ART_RADIAL];
	double			dest_ArtCoeff[ANGLE][CAMNUM][ART_ANGULAR][ART_RADIAL];
	double			MinErr, err;
	int				align[60][CAMNUM_2];
	unsigned char	q8_ArtCoeff[ANGLE][CAMNUM][ART_COEF];
	unsigned char	q4_ArtCoeff[ANGLE][CAMNUM][ART_COEF_2];
	// for color decsriptor
	uint64_t CompactColor[ANGLE][CAMNUM];	// 63 bits for each image
	uint64_t dest_CompactColor[ANGLE][CAMNUM];	// 63 bits for each image
	// for circularity
	double			cir_Coeff[ANGLE][CAMNUM];
	unsigned char	q8_cirCoeff[ANGLE][CAMNUM], dest_cirCoeff[ANGLE][CAMNUM];
	// for fourier descriptor
	double			src_FdCoeff[ANGLE][CAMNUM][FD_COEFF_NO], dest_FdCoeff[ANGLE][CAMNUM][FD_COEFF_NO];
	unsigned char	q8_FdCoeff[ANGLE][CAMNUM][FD_COEFF_NO];
	sPOINT			*Contour;
	unsigned char	*ContourMask;
	// for eccentricity
	double			ecc_Coeff[ANGLE][CAMNUM];
	unsigned char	q8_eccCoeff[ANGLE][CAMNUM], dest_eccCoeff[ANGLE][CAMNUM];
	// for compare
	int				TopNum;
	pMatRes			pSearch, pmr, pmrr, pTop;
 	int				high, low, middle;
	double			QuantTable[17] = {	0.000000000, 0.003585473, 0.007418411, 0.011535520, 
										0.015982337, 0.020816302, 0.026111312, 0.031964674, 
										0.038508176, 0.045926586, 0.054490513, 0.064619488, 
										0.077016351, 0.092998687, 0.115524524, 0.154032694, 1.000000000};
	double			CenX[CAMNUM], CenY[CAMNUM];
	int				total;

  fpt_art_q4 = fopen("all_q4_v1.8.art", "wb");
  fpt_art_q8 = fopen("all_q8_v1.8.art", "wb");
//		fpt_ccd = fopen("all_v1.7.ccd", "wb");
  fpt_cir_q8 = fopen("all_q8_v1.8.cir", "wb");
//		fpt_fd = fopen("all.fd", "wb");
  fpt_fd_q8 = fopen("all_q8_v1.8.fd", "wb");
  fpt_ecc_q8 = fopen("all_q8_v1.8.ecc", "wb");
  // initialize ART
  GenerateBasisLUT();
  // initialize: read camera set
  for(destCam=0; destCam<ANGLE; destCam++)
  {
    sprintf(filename, "12_%d", destCam);
    ReadObj(filename, CamVertex+destCam, CamTriangle+destCam, CamNumVer+destCam, CamNumTri+destCam);
  }

  for(i=0; i<CAMNUM; i++)
  {
    srcBuff[i] = (unsigned char *) malloc (winw * winh * sizeof(unsigned char));
    ColorBuff[i] = (unsigned char *) malloc (3 * winw * winh * sizeof(unsigned char));
  }
  YuvBuff = (unsigned char *) malloc (3 * winw * winh * sizeof(unsigned char));
  // add edge to test retrieval
  EdgeBuff = (unsigned char *) malloc (winw * winh * sizeof(unsigned char));

  // for Fourier Descriptor
  total = winw * winh;
  Contour = (sPOINT *) malloc( total * sizeof(sPOINT));
  ContourMask = (unsigned char *) malloc( total * sizeof(unsigned char));

  // get the translatation and scale of the two model
  if( ReadObj(fname, &vertex1, &triangle1, &NumVer1, &NumTri1) == 0 ) {
    exit(1);
  }

  // ****************************************************************
  // Corase alignment
  // ****************************************************************

  // Translate and scale model 1
  TranslateScale(vertex1, NumVer1, triangle1, NumTri1, fname, &Translate1, &Scale1);

  // read RED only, so size is winw*winh
  for(srcCam=0; srcCam<ANGLE; srcCam++)
  {
    // capture CAMNUM silhouette of srcfn to memory
    for(i=0; i<CAMNUM; i++) {
//					RenderToMem(srcBuff[i], ColorBuff[i], CamVertex[srcCam]+i, vertex1, triangle1, NumVer1, NumTri1);
      RenderToMem(srcBuff[i], NULL, CamVertex[srcCam]+i, vertex1, triangle1, NumVer1, NumTri1);
    }

    // find center for each shape
    for(i=0; i<CAMNUM; i++)
      FindCenter(srcBuff[i], winw, winh, CenX+i, CenY+i);


    // get Zernike moment
    FindRadius(srcBuff, CenX, CenY);
    for(i=0; i<CAMNUM; i++)
    {
      ExtractCoefficients(srcBuff[i], src_ArtCoeff[srcCam][i], EdgeBuff, CenX[i], CenY[i]);
    }

    // get Fourier descriptor
    for(i=0; i<CAMNUM; i++)
      FourierDescriptor(src_FdCoeff[srcCam][i], srcBuff[i], winw, winh, Contour, ContourMask, CenX[i], CenY[i]);

    // get eccentricity
    for(i=0; i<CAMNUM; i++)
      ecc_Coeff[srcCam][i] = Eccentricity(srcBuff[i], winw, winh, CenX[i], CenY[i]);

    // get circularity
    for(i=0; i<CAMNUM; i++)
    {
      EdgeDetectSil(EdgeBuff, srcBuff[i], winw, winh);
      cir_Coeff[srcCam][i] = Circularity(srcBuff[i], winw, winh, EdgeBuff);
    }

  }

  // free memory of 3D model
  free(vertex1);
  free(triangle1);



  // linear Quantization to 8 bits for each coefficient
  for(i=0; i<ANGLE; i++)
    for(j=0; j<CAMNUM; j++)
    {
      // the order is the same with that defined in MPEG-7, total 35 coefficients
      k = 0;
      p = 0;
      for(r=1 ; r<ART_RADIAL ; r++, k++)
      {
        itmp = (int)(QUANT8 *  src_ArtCoeff[i][j][p][r]);
        if(itmp>255)
          q8_ArtCoeff[i][j][k] = 255;
        else
          q8_ArtCoeff[i][j][k] = itmp;
      }

      for(p=1; p<ART_ANGULAR ; p++)
        for(r=0 ; r<ART_RADIAL ; r++, k++)
        {
          itmp = (int)(QUANT8 *  src_ArtCoeff[i][j][p][r]);
          if(itmp>255)
            q8_ArtCoeff[i][j][k] = 255;
          else
            q8_ArtCoeff[i][j][k] = itmp;
        }
    }

  // save to disk
  fwrite(q8_ArtCoeff, sizeof(unsigned char), ANGLE * CAMNUM * ART_COEF, fpt_art_q8);
  sprintf(filename, "%s_q8_v1.8.art", fname);
  if( (fpt = fopen(filename, "wb")) == NULL )	{	printf("Write %s error!!\n", filename);	return;	}
  fwrite(q8_ArtCoeff, sizeof(unsigned char), ANGLE * CAMNUM * ART_COEF, fpt);
  fclose(fpt);

  // non-linear Quantization to 4 bits for each coefficient using MPEG-7 quantization table
  for(i=0; i<ANGLE; i++)
    for(j=0; j<CAMNUM; j++)
    {
      // the order is the same with that defined in MPEG-7, total 35 coefficients
      k = 0;
      p = 0;
      for(r=1 ; r<ART_RADIAL ; r++, k++)
      {
        high = 17;
        low = 0;
        while(high-low > 1)
        {
          middle = (high+low) / 2;

          if(QuantTable[middle] < src_ArtCoeff[i][j][p][r])
            low = middle;
          else
            high = middle;
        }
        q8_ArtCoeff[i][j][k] = low;
      }
      for(p=1; p<ART_ANGULAR ; p++)
        for(r=0 ; r<ART_RADIAL ; r++, k++)
        {
          high = 17;
          low = 0;
          while(high-low > 1)
          {
            middle = (high+low) / 2;

            if(QuantTable[middle] < src_ArtCoeff[i][j][p][r])
              low = middle;
            else
              high = middle;
          }
          q8_ArtCoeff[i][j][k] = low;
        }
    }

  for(i=0; i<ANGLE; i++)
    for(j=0; j<CAMNUM; j++)
      for(k=0, a=0; k<ART_COEF; k+=2, a++)
        if( k+1 < ART_COEF )
          q4_ArtCoeff[i][j][a] = ( (q8_ArtCoeff[i][j][k] << 4) & 0xf0 ) | 
                    ( q8_ArtCoeff[i][j][k+1] & 0x0f );
        else
          q4_ArtCoeff[i][j][a] = ( (q8_ArtCoeff[i][j][k] << 4) & 0xf0 );

  // save to disk
  fwrite(q4_ArtCoeff, sizeof(unsigned char), ANGLE * CAMNUM * ART_COEF_2, fpt_art_q4);
  sprintf(filename, "%s_q4_v1.8.art", fname);
  if( (fpt = fopen(filename, "wb")) == NULL )	{	printf("Write %s error!!\n", filename);	return;	}
  fwrite(q4_ArtCoeff, sizeof(unsigned char), ANGLE * CAMNUM * ART_COEF_2, fpt);
  fclose(fpt);


  // linear Quantization to 8 bits for each coefficient
  for(i=0; i<ANGLE; i++)
    for(j=0; j<CAMNUM; j++)
    {
      itmp = (int)(QUANT8 *  cir_Coeff[i][j]);
      if(itmp>255)		q8_cirCoeff[i][j] = 255;
      else				q8_cirCoeff[i][j] = itmp;
    }
  // save to disk
  fwrite(q8_cirCoeff, sizeof(unsigned char), ANGLE * CAMNUM, fpt_cir_q8);
  sprintf(filename, "%s_q8_v1.8.cir", fname);
  if( (fpt = fopen(filename, "wb")) == NULL )	{	printf("Write %s error!!\n", filename);	return;	}
  fwrite(q8_cirCoeff, sizeof(unsigned char), ANGLE * CAMNUM, fpt);
  fclose(fpt);

  // **********************************************************************
  // save eccentricity feature to file
  // linear Quantization to 8 bits for each coefficient
  for(i=0; i<ANGLE; i++)
    for(j=0; j<CAMNUM; j++)
    {
      itmp = (int)(QUANT8 * ecc_Coeff[i][j]);
      if(itmp>255)		q8_eccCoeff[i][j] = 255;
      else				q8_eccCoeff[i][j] = itmp;
    }
  // save to disk
  fwrite(q8_eccCoeff, sizeof(unsigned char), ANGLE * CAMNUM, fpt_ecc_q8);
  sprintf(filename, "%s_q8_v1.8.ecc", fname);
  if( (fpt = fopen(filename, "wb")) == NULL )	{	printf("Write %s error!!\n", filename);	return;	}
  fwrite(q8_eccCoeff, sizeof(unsigned char), ANGLE * CAMNUM, fpt);
  fclose(fpt);

  // **********************************************************************

  for(i=0; i<ANGLE; i++)
    for(j=0; j<CAMNUM; j++)
    {
      for(k=0; k<FD_COEFF_NO; k++)
      {
        itmp = (int)(QUANT8 * FD_SCALE * src_FdCoeff[i][j][k]);
        if(itmp>255)
          q8_FdCoeff[i][j][k] = 255;
        else
          q8_FdCoeff[i][j][k] = itmp;
      }
    }

  fwrite(q8_FdCoeff, ANGLE * CAMNUM * FD_COEFF_NO, sizeof(unsigned char), fpt_fd_q8);
  sprintf(filename, "%s_q8_v1.8.fd", fname);
  fpt = fopen(filename, "wb");
  fwrite(q8_FdCoeff, ANGLE * CAMNUM * FD_COEFF_NO, sizeof(unsigned char), fpt);
  fclose(fpt);

  for(i=0; i<CAMNUM; i++)
  {
    free(srcBuff[i]);
    free(ColorBuff[i]);
  }
  free(YuvBuff);
  free(EdgeBuff);
  free(Contour);
  free(ContourMask);
  fclose(fpt_art_q8);
  fclose(fpt_art_q4);
//		fclose(fpt_ccd);
  fclose(fpt_cir_q8);
  fclose(fpt_ecc_q8);
//		fclose(fpt_fd);
  fclose(fpt_fd_q8);
  for(destCam=0; destCam<ANGLE; destCam++)
  {
    free(CamVertex[destCam]);
    free(CamTriangle[destCam]);
  }
  NumTri = 0;
  exit(0);
}

void init(void) 
{
	glClearColor (1.0, 1.0, 1.0, 0.0);
	glClearDepth(1.0);
	glEnable(GL_DEPTH_TEST);
}

void reshape (int w, int h)
{
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	glOrtho(-1, 1, -1, 1, 0.0, 2.0);
	glViewport (0, 0, (GLsizei) winw, (GLsizei) winh); 

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(1,0,0,0,0,0,0,1,0);
}

void printHelp() {
  printf("Usage: \n");
  printf("\t ./3DAlignment obj\n");
  printf("obj - name of the object to be encoded\n");
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("ERROR: The program has only 1 argument, received %d instead!\n",
           argc - 1);
    printHelp();
    exit(2);
  }
  fname = argv[1];

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(WIDTH, HEIGHT); 
	glutInitWindowPosition(100, 100);
	glutCreateWindow (argv[0]);
  glutPushWindow();
  glutHideWindow();
	init();
	glutDisplayFunc(display); 
	glutReshapeFunc(reshape);
  glutIdleFunc(renderingFun);
  glutMainLoop();


	return 0;
}
