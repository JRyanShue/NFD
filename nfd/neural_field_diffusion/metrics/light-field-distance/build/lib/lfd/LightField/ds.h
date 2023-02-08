// for matching each model
typedef struct MatRes_ *pMatRes;
typedef struct MatRes_ {				// Match Result
	char			name[200];			// file name
//	double			sim;	
	int				sim;				// similarity of this file
//	float				sim;				// similarity of this file
	int				index;
//	pMatRes			pointer;			// poiner to next file
	float			Precise;
	int				Recall;
}MatRes;

#ifndef	PARA_
#define PARA_
// #1
	#define	WIDTH			256
	#define HEIGHT			256
//	#define	TOTAL_PIXEL		65025	// 255x255 (WIDTH*HEIGHT)
	#define ANGLE			10		// for dest
	#define SRC_ANGLE		10
	#define CAMNUM			10
	#define CAMNUM_2		20

	#define CENTER_X		128		// WIN_WIDTH/2
	#define CENTER_Y		128		// WIN_HEIGHT/2
	#define ART_ANGULAR		12
	#define ART_RADIAL 		3
	#define ART_COEF 		35//36
	#define ART_COEF_q4 	18
	#define	ART_LUT_RADIUS	50		// Zernike basis function radius
	#define	ART_LUT_SIZE	101		// (ART_LUT_RADIUS*2+1)
	#define PI				3.141592653
	#define HYPOT(x,y)		sqrt((x)*(x)+(y)*(y))
	#define	FD_COEF	10

// #2
	#define	SPHERE_NUM	64
	#define HAMONIC_NUM	64
	#define	SH_KEY_NUM	SPHERE_NUM * HAMONIC_NUM

// #3
	#define S3D_KEY_NUM		22

#endif

