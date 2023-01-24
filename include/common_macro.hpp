#pragma once

#ifndef COMMON_HPP
#define COMMON_HPP


#define SCALAR_PRODUCT_2p(vec1, vec2) ( ((vec1(0))*((vec2)(0)) + (vec1(1))*((vec2)(1)) + (vec1(2))*((vec2)(2))) )
#define NUM2SQ(x) ((x)*(x))

//#define MIN(a, b) (a<b ? a : b)
//#define MAX(a, b) (a>b ? a : b)
#define MALLOC_(type,n) (type *)malloc((n)*sizeof(type))

#define UNKNOWN_CELL_LABEL 2
#define NOT_TRAV_CELL_LABEL  -1
#define TRAV_CELL_LABEL  1

#define MIN_NUM_POINTS_IN_CELL 2

#define LOWEST_Z_VALUE -100.f
#define HIGHEST_Z_VALUE 100.0f


#define IS_ROAD(l)            ( (l) == 40 )
#define IS_SIDEWALK(l)        ( (l) == 48 )
#define IS_PARKING(l)         ( (l) == 44 )
#define IS_OTHER_GROUND(l)    ( (l) == 49 )
#define IS_LANE_MARKING(l)    ( (l) == 60 )

#define POINTBELONGSTOROAD(l) ( ( \
                    IS_ROAD           ( (l) )  \
                    || IS_PARKING     ( (l) )  \
                    || IS_OTHER_GROUND( (l) )  \
                    || IS_LANE_MARKING( (l) )  \
                    || IS_SIDEWALK    ( (l) ) \
                    ) )
                    
                    //|| IS_SIDEWALK    ( (l) ) 

#define LABELED(l) ( (l) > 1 )




#define IS_DRIVEABLE_SURFACE_NU(l)            ( (l) == 24 )
#define IS_OTHER_NU(l)        ( (l) == 25 )
#define IS_SIDEWALK_NU(l)        ( (l) == 26 )
#define IS_TERRAIN_NU(l) ((l) == 27)

#define POINTBELONGSTOROAD_NU(l) ( (l)>=24 && (l)<=27 )

#define LABELED_NU(l) ( (l) > 0 )


#endif