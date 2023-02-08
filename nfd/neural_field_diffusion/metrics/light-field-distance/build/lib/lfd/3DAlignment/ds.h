#ifndef	MRG_PARA
#define MRG_PARA
	#define		RNUM		7		// degree of multi-resolution
	#define		MU_RANGE	64		// RNUM=7 => 64, 32, 16, 8, 4, 2, 1
	#define		RANGENUM	127		// how many range numbers are used ( 2^7 - 1 )
#endif


// a linker list
typedef struct RefIndex_ * pRefIndex;
typedef struct RefIndex_ {
	int				index;			// triangle index
	pRefIndex		pointer;
}RefIndex;

// for Short-Cut Edges
typedef struct SCEdge_ *pSCEdge;
typedef struct SCEdge_ {
	int				VerIndex;
	double			dist;
	pSCEdge			pointer;
}SCEdge;

// for Different Parts Edges, which connect different parts in a model
typedef struct DPEdge_ *pDPEdge;
typedef struct DPEdge_ {
	int				VerIndex;
	double			dist;
	pDPEdge			pointer;
}DPEdge;

// data structure of vertex
typedef struct Ver_ * pVer;
typedef struct Ver_ {
	double			coor[3];		// coordinate coor[0], coor[1], coor[2]
//	double			mu;
//	int				range;			// this is mu_range number, i.e., 0~63
//	pSCEdge			sce;			// short-cut edge
//	pDPEdge			dpe;			// different-part edge
//	pRefIndex		pointer;		// used for recording which triangles use the vertex
									// goal: easily geting connected component of triangle
}Ver;

// data structure of triangle
typedef struct Tri_ * pTri;
typedef struct Tri_ {
	int				v[15];			// map to vertex 0, 1, 2
	int				NodeName;		// node index
//	double			area;
//	pTri			pointer;		// save the split trianlge
	double			r, g, b;		// color of this triangle
}Tri;

// the meta data format of each Reeb Graph node, used for matching different models
typedef struct Node_ * pNode;
typedef struct Node_ {
	int				name;
	pRefIndex		child, R_edge;
	int				parent;
	int				range;
	double			area, length;
}Node;

// VLIST for Dijkstra's algorithm (heap insert and extract)
typedef struct VLIST_ *pVLIST;
typedef struct VLIST_ {
	int				VerIndex;
	double			dist;
	int				visit;		// to mark if the vertex has visited
}VLIST;

// for base vertex of Geodesic distance
typedef struct BLIST_ *pBLIST;
typedef struct BLIST_ {
	int				VerIndex;
	double			area;
	pBLIST			pointer;
}BLIST;

// data structure of new vertex for MRG, first save as linker list
typedef struct NewVerP_ * pNewVerP;
typedef struct NewVerP_ {
	int				VerName;
	double			coor[3];			// coordinate r[0], r[1], r[2]
	double			mu;
	int				fromVer, toVer;		// the new vertex is generate between "fromVer" and "toVer", "fromVer" < "toVer"
	pRefIndex		tri;				// used for recording which triangles use the vertex
										// goal: easily geting connected component of triangle
	pNewVerP		pointer;
}NewVerP;

// MLIST for MRG
typedef struct ALIST_ *pALIST;
typedef struct ALIST_ {
	pTri			tri;
	pALIST			pointer;
}ALIST;

// for matching each model
typedef struct MatRes_ *pMatRes;
typedef struct MatRes_ {				// Match Result
	char			name[100];			// file name
	double			sim;				// similarity of this file
//	int				index;
	pMatRes			pointer;			// poiner to next file
}MatRes;

// for merging different parts, save quantization value
typedef struct Quant_ *pQuant;
typedef struct Quant_ {
	int				coor[3];		// coordinate coor[0], coor[1], coor[2] of quantization version
	int				index;
}Quant;

// for resampling, record index and coordinate of each vertex
typedef struct ResamVer_ *pResamVer;
typedef struct ResamVer_ {
	int				index;
	double			coor[3];		// coordinate coor[0], coor[1], coor[2]
}ResamVer;

// for resampling, record how many new vertex ( "num" ) are added in this edge ( connect to "index" )
typedef struct ResamEdge_ *pResamEdge;
typedef struct ResamEdge_ {
	int				index;		// the vertex connect to "index"
	int				num;		// the edge split by "num" vertices
	pResamVer		NewVer;		// point to coordinate and index of those vertex
	pResamEdge		pointer;
}ResamEdge;

// when we match R-node, also take adjacent R-nodes (i.e. the graph stracture) into account
typedef struct AdjRnode_ *pAdjRnode;
typedef struct AdjRnode_ {
	double			area, length;
}AdjRnode;

typedef struct vector_
{
	double x, y, z;
}vector;

typedef struct Meterial_ *pMeterial;
typedef struct Meterial_ {
	char			name[100];
	double			r, g, b;
	pMeterial		pointer;
}Meterial;

#ifndef	PARA_
#define PARA_
	#define	WIDTH			256
	#define HEIGHT			256
//	#define	TOTAL_PIXEL		65025	// 255x255 (WIDTH*HEIGHT)
	#define ANGLE			10		// for dest
	#define CAMNUM			10
	#define CAMNUM_2		20

	#define CENTER_X		127.5	//128		// WIN_WIDTH/2
	#define CENTER_Y		127.5	//128		// WIN_HEIGHT/2
	#define ART_ANGULAR		12
	#define ART_RADIAL 		3
	#define ART_COEF 		35//36
	#define ART_COEF_2 		18
	#define	ART_LUT_RADIUS	50		// Zernike basis function radius
	#define	ART_LUT_SIZE	101		// (ART_LUT_RADIUS*2+1)
	#define PI				3.141592653

	// circularity and eccentricity (from MPEG-7 XM)
	#define CIR_MIN          12.0
	#define CIR_MAX          110.0
	#define CIR_RANGE		 98.0
	#define ECC_MIN          1.0
	#define ECC_MAX          10.0
	#define ECC_RANGE	  	 9.0

	#include <math.h>
	#define HYPOT(x,y)		sqrt((x)*(x)+(y)*(y))

// for color bins
	#define		NUM_BINS	64

// for fourier descriptor
	#define		FD_COEFF_NO	10

	typedef struct sPOINT_
	{	
		short int	x, y;
	}sPOINT;

#endif

