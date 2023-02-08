#include <math.h>
#include <memory.h>
#include "ds.h"

// extract feature:
	// Initial call                     : GenerateBasisLUT()
	// then			                    : FindRadius()
	// then for each silhouette         : ExtractCoefficients()
// compare:
	// call                             : GetDistance()

static double	m_pBasisR[ART_ANGULAR][ART_RADIAL][ART_LUT_SIZE][ART_LUT_SIZE];	// real-value of RegionShape basis function
static double	m_pBasisI[ART_ANGULAR][ART_RADIAL][ART_LUT_SIZE][ART_LUT_SIZE];	// imaginary-value of RegionShape basis function

double			m_radius;
double			r_radius;

/*
double GetReal(int p, int r, double dx, double dy)
{
	int x = (int)dx;
	int y = (int)dy;

	double ix = dx - x;
	double iy = dy - y;

	double x1 = m_pBasisR[p][r][x][y] + (m_pBasisR[p][r][x+1][y]-m_pBasisR[p][r][x][y]) * ix;
	double x2 = m_pBasisR[p][r][x][y+1] + (m_pBasisR[p][r][x+1][y+1]-m_pBasisR[p][r][x][y+1]) * ix;

	return (x1 + (x2-x1) * iy);
}

double GetImg(int p, int r, double dx, double dy)
{
	int x = (int)dx;
	int y = (int)dy;

	double ix = dx - x;
	double iy = dy - y;

	double x1 = m_pBasisI[p][r][x][y] + (m_pBasisI[p][r][x+1][y]-m_pBasisI[p][r][x][y]) * ix;
	double x2 = m_pBasisI[p][r][x][y+1] + (m_pBasisI[p][r][x+1][y+1]-m_pBasisI[p][r][x][y+1]) * ix;

	return (x1 + (x2-x1) * iy);
}
*/

void ExtractCoefficients(unsigned char *Y, double m_Coeff[ART_ANGULAR][ART_RADIAL], unsigned char *Edge, double CenX, double CenY)
{
	int				x, y, ix, iy;
	int				p, r;
	double			dx, dy, tx, ty, x1, x2;
	int				count;
	double			m_pCoeffR[ART_ANGULAR][ART_RADIAL];
	double			m_pCoeffI[ART_ANGULAR][ART_RADIAL];
//	double			norm;

	unsigned char *pImage;
//	unsigned char *pEdge;

	memset(m_pCoeffR, 0, ART_ANGULAR * ART_RADIAL * sizeof(double) );
	memset(m_pCoeffI, 0, ART_ANGULAR * ART_RADIAL * sizeof(double) );
//	for(p=0 ; p<ART_ANGULAR ; p++)
//	for(r=0 ; r<ART_RADIAL ; r++)
//	{
//		m_pCoeffR[p][r] = 0;
//		m_pCoeffI[p][r] = 0;
//	}

	count = 0;
	pImage = Y;
//	pEdge = Edge;
	for (y=0 ; y<HEIGHT ; y++)
	for (x=0 ; x<WIDTH; x++)
	{
//		if( *pImage < 127 )
		if( *pImage < 255 )
		{
			// 1.0 is for silhouette, another one is for depth, both weighting is 0.5
			// both depth and silhouette
//			norm = (1.0 + (255.0-*pImage) / 255.0) / 2;
			// depth only
//			norm = (255.0-*pImage) / 255.0;
			// silhouette only
//			norm = 1.0;
			// edge (from depth) only
//			norm = *pEdge / 255.0;
			// edge (from depth) + depth + silhouette
//			norm = (*pEdge/255.0 + (255.0-*pImage)/255.0 + 1.0) / 3;
			// edge (from silhouette) only
//			norm = 1.0;
			// edge (from silhouette) + silhouette
//			norm = (255.0-*pImage) / 255.0;

			// map image coordinate (x,y) to basis function coordinate (tx,ty)
//			dx = x - CENTER_X;
//			dy = y - CENTER_Y;
			dx = x - CenX;
			dy = y - CenY;
			tx = dx * r_radius + ART_LUT_RADIUS;
			ty = dy * r_radius + ART_LUT_RADIUS;
			ix = (int)tx;
			iy = (int)ty;
			dx = tx - ix;
			dy = ty - iy;

			// summation of basis function
//			if(tx >= 0 && tx < ART_LUT_SIZE && ty >= 0 && ty < ART_LUT_SIZE)
			for(p=0 ; p<ART_ANGULAR ; p++)
			for(r=0 ; r<ART_RADIAL ; r++)
			{
				// GetReal (if call function, the speed will be very slow)
				// m_pCoeffR[p][r] += GetReal(p, r, tx, ty);
				x1 = m_pBasisR[p][r][ix][iy] + (m_pBasisR[p][r][ix+1][iy]-m_pBasisR[p][r][ix][iy]) * dx;
				x2 = m_pBasisR[p][r][ix][iy+1] + (m_pBasisR[p][r][ix+1][iy+1]-m_pBasisR[p][r][ix][iy+1]) * dx;
//				m_pCoeffR[p][r] += norm * (x1 + (x2-x1) * dy);
				m_pCoeffR[p][r] += (x1 + (x2-x1) * dy);

				// GetImg (if call function, the speed will be very slow)
				// m_pCoeffI[p][r] -= GetImg(p, r, tx, ty);
				x1 = m_pBasisI[p][r][ix][iy] + (m_pBasisI[p][r][ix+1][iy]-m_pBasisI[p][r][ix][iy]) * dx;
				x2 = m_pBasisI[p][r][ix][iy+1] + (m_pBasisI[p][r][ix+1][iy+1]-m_pBasisI[p][r][ix][iy+1]) * dx;
//				m_pCoeffI[p][r] -= norm * (x1 + (x2-x1) * dy);
				m_pCoeffI[p][r] -= (x1 + (x2-x1) * dy);
			}

			count ++;		// how many pixels
		}
		pImage++;
//		pEdge++;
	}

	// if the 3D model is flat, some camera will render nothing, so count=0 in this case
	if( count > 0 )
	{
		for(p=0 ; p<ART_ANGULAR ; p++)
		for(r=0 ; r<ART_RADIAL ; r++)
			m_Coeff[p][r] = HYPOT( m_pCoeffR[p][r]/count, m_pCoeffI[p][r]/count );

		// normalization
		for(p=0 ; p<ART_ANGULAR ; p++)
		for(r=0 ; r<ART_RADIAL ; r++)
			m_Coeff[p][r] /= m_Coeff[0][0];
			
	}
	else
	{
		// if didn't add this, the result will also be saved as 0
		for(p=0 ; p<ART_ANGULAR ; p++)
		for(r=0 ; r<ART_RADIAL ; r++)
			m_Coeff[p][r] = 0.0;
		// use a line to test the number to approximate
/*		for(p=0 ; p<ART_ANGULAR ; p++)
		for(r=0 ; r<ART_RADIAL ; r++)
			m_Coeff[p][r] = 0.010256410;
		for(p=0 ; p<ART_ANGULAR ; p+=2)
			m_Coeff[p][0] = 0.980129780;*/
	}
}

// speed up this function later
void FindRadius(unsigned char *Y[CAMNUM], double *CenX, double *CenY)
{
	int				x, y;
	double			temp_radius;
	unsigned char	*pImage;
	int				i;

	// Find maximum radius from center of mass
	m_radius = 0;
	for(i=0; i<CAMNUM; i++)
	{
		pImage = Y[i];
		for(y=0 ; y<HEIGHT; y++)
			for(x=0 ; x<WIDTH ; x++)
			{
//				if( *pImage < 127 )
				if( *pImage < 255 )
				{
//					temp_radius = HYPOT(x - CENTER_X, y - CENTER_Y);
					temp_radius = HYPOT(x - CenX[i], y - CenY[i]);
					if(temp_radius > m_radius)
						m_radius = temp_radius;
				}
				pImage++;
			}
	}

	r_radius = ART_LUT_RADIUS / m_radius;
}

void GenerateBasisLUT()
{
	double	angle, temp, radius;
	int		p, r, x, y;
	int		maxradius;

	maxradius = ART_LUT_RADIUS;

	for(y=0 ; y<ART_LUT_SIZE ; y++)
	for(x=0 ; x<ART_LUT_SIZE ; x++)
	{
		radius = HYPOT(x-maxradius, y-maxradius);
		if(radius < maxradius)
		{
			angle = atan2(y-maxradius, x-maxradius);

			for(p=0 ; p<ART_ANGULAR ; p++)
			for(r=0 ; r<ART_RADIAL ; r++)
			{
				temp = cos(radius*PI*r/maxradius);
				m_pBasisR[p][r][x][y] = temp*cos(angle*p);
				m_pBasisI[p][r][x][y] = temp*sin(angle*p);
			}
		}
		else
		{
			for(p=0 ; p<ART_ANGULAR ; p++)
			for(r=0 ; r<ART_RADIAL ; r++)
			{
				m_pBasisR[p][r][x][y] = 0;
				m_pBasisI[p][r][x][y] = 0;
			}
		}
	}
}

double GetDistance(double m_Coeff1[ART_ANGULAR][ART_RADIAL], double m_Coeff2[ART_ANGULAR][ART_RADIAL])
{
	// perform matching
	double	distance;
	int		i, j;

	distance = 0;
	for(i=0 ; i<ART_ANGULAR ; i++)
	for(j=0 ; j<ART_RADIAL ; j++)
		if(i!=0 || j!=0)			// can I delete this condition??
			distance += fabs( m_Coeff1[i][j] - m_Coeff2[i][j] );

	return distance;
}
