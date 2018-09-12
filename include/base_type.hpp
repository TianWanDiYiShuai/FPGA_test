#ifndef BASE_TYPE_HPP
#define BASE_TYPE_HPP

#include <stdint.h>
#ifndef NDEBUG
#include <assert.h>
#endif

typedef int8_t		i8;
typedef uint8_t		u8;
typedef int16_t		i16;
typedef uint16_t	u16;
typedef int32_t		i32;
typedef uint32_t	u32;
typedef int64_t		i64;
typedef uint64_t	u64;


#ifndef __cplusplus
/*enum {
false	= 0,
true	= 1
};
typedef _Bool           bool;*/
typedef enum {
	false = 0,
	true = 1
} bool;

enum {
	OK = 0,
	NOK = -1
};
#endif

/* ASSERT */
#ifndef NDEBUG
#define ASSERT(x) assert(x)
#else
#define ASSERT(x)
#endif

/* TEST_DATA */
#ifndef TEST_DATA
#define COMMENT(b,...)
#else
#define COMMENT(b,...)	if ((b)->stream_trace) { \
	char buffer[128]; \
	snprintf(buffer, sizeof(buffer), ##__VA_ARGS__); \
	ASSERT(strlen((b)->stream_trace->comment) + strlen(buffer) < \
	sizeof((b)->stream_trace->comment)); \
	strcat((b)->stream_trace->comment, buffer); \
}
#endif

/* General tools */
#define ABS(x)		((x) < (0) ? -(x) : (x))
#define MAX(a, b)	((a) > (b) ?  (a) : (b))
#define MIN(a, b)	((a) < (b) ?  (a) : (b))
#define SIGN(a)		((a) < (0) ? (-1) : (1))
#define CLIP3(x, y, z)	((z) < (x) ? (x) : ((z) > (y) ? (y) : (z)))

#endif
