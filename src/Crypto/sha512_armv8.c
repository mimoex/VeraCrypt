#include "Common/Tcdefs.h"
#include "config.h"
#include <stdint.h>

#if CRYPTOPP_BOOL_ARM64

#include <arm_neon.h>

#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("sha3"))), apply_to=function)
#endif

CRYPTOPP_ALIGN_DATA(64) static const uint64_t K[80] = {
	0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL,
	0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
	0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
	0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
	0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
	0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
	0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL,
	0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
	0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
	0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
	0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL,
	0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
	0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL,
	0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
	0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
	0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
	0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL,
	0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
	0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL,
	0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
	0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
	0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
	0xd192e819d6ef5218ULL, 0xd69906245565a910ULL,
	0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
	0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
	0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
	0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
	0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
	0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL,
	0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
	0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL,
	0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
	0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
	0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
	0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
	0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
	0x28db77f523047d84ULL, 0x32caab7b40c72493ULL,
	0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
	0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
	0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

static inline uint64x2_t load_input(const uint8_t *p)
{
	return vreinterpretq_u64_u8(
		vrev64q_u8(vld1q_u8(p)));
}

static inline uint64x2_t schedule_update(
	uint64x2_t m8,
	uint64x2_t m7,
	uint64x2_t m4,
	uint64x2_t m3,
	uint64x2_t m1)
{
	return vsha512su1q_u64(
		vsha512su0q_u64(m8, m7),
		m1,
		vextq_u64(m4, m3, 1));
}

static inline void round2(
	unsigned int round,
	uint64x2_t w,
	uint64x2_t *ab,
	uint64x2_t *cd,
	uint64x2_t *ef,
	uint64x2_t *gh)
{
	const uint64x2_t k = vld1q_u64(&K[round]);

	uint64x2_t wk = vaddq_u64(w, k);
	wk = vextq_u64(wk, wk, 1);

	const uint64x2_t sum = vaddq_u64(wk, *gh);
	const uint64x2_t de = vextq_u64(*cd, *ef, 1);
	const uint64x2_t fg = vextq_u64(*ef, *gh, 1);

	const uint64x2_t t = vsha512hq_u64(sum, fg, de);

	*gh = vsha512h2q_u64(t, *cd, *ab);
	*cd = vaddq_u64(*cd, t);
}

void sha512_compress_digest_armv8(
	const void *input_data,
	uint64_t digest[8],
	uint64_t num_blks)
{
	const uint8_t *p = (const uint8_t *) input_data;

	uint64x2_t state_ab = vld1q_u64(&digest[0]);
	uint64x2_t state_cd = vld1q_u64(&digest[2]);
	uint64x2_t state_ef = vld1q_u64(&digest[4]);
	uint64x2_t state_gh = vld1q_u64(&digest[6]);

	while (num_blks--)
	{
		const uint64x2_t save_ab = state_ab;
		const uint64x2_t save_cd = state_cd;
		const uint64x2_t save_ef = state_ef;
		const uint64x2_t save_gh = state_gh;

		uint64x2_t ab = state_ab;
		uint64x2_t cd = state_cd;
		uint64x2_t ef = state_ef;
		uint64x2_t gh = state_gh;

		uint64x2_t s0 = load_input(p + 16 * 0);
		uint64x2_t s1 = load_input(p + 16 * 1);
		uint64x2_t s2 = load_input(p + 16 * 2);
		uint64x2_t s3 = load_input(p + 16 * 3);
		uint64x2_t s4 = load_input(p + 16 * 4);
		uint64x2_t s5 = load_input(p + 16 * 5);
		uint64x2_t s6 = load_input(p + 16 * 6);
		uint64x2_t s7 = load_input(p + 16 * 7);

#define R2(N, S, A, C, E, G) round2((N), (S), &(A), &(C), &(E), &(G))

		R2( 0, s0, ab, cd, ef, gh);
		R2( 2, s1, gh, ab, cd, ef);
		R2( 4, s2, ef, gh, ab, cd);
		R2( 6, s3, cd, ef, gh, ab);
		R2( 8, s4, ab, cd, ef, gh);
		R2(10, s5, gh, ab, cd, ef);
		R2(12, s6, ef, gh, ab, cd);
		R2(14, s7, cd, ef, gh, ab);

#define UPDATE_AND_ROUND(N, S0, S1, S4, S3, S7, A, C, E, G) \
	do { \
		(S0) = schedule_update((S0), (S1), (S4), (S3), (S7)); \
		R2((N), (S0), (A), (C), (E), (G)); \
	} while (0)

		for (unsigned int base = 16; base < 80; base += 16)
		{
			UPDATE_AND_ROUND(base +  0, s0, s1, s4, s5, s7, ab, cd, ef, gh);
			UPDATE_AND_ROUND(base +  2, s1, s2, s5, s6, s0, gh, ab, cd, ef);
			UPDATE_AND_ROUND(base +  4, s2, s3, s6, s7, s1, ef, gh, ab, cd);
			UPDATE_AND_ROUND(base +  6, s3, s4, s7, s0, s2, cd, ef, gh, ab);
			UPDATE_AND_ROUND(base +  8, s4, s5, s0, s1, s3, ab, cd, ef, gh);
			UPDATE_AND_ROUND(base + 10, s5, s6, s1, s2, s4, gh, ab, cd, ef);
			UPDATE_AND_ROUND(base + 12, s6, s7, s2, s3, s5, ef, gh, ab, cd);
			UPDATE_AND_ROUND(base + 14, s7, s0, s3, s4, s6, cd, ef, gh, ab);
		}

#undef UPDATE_AND_ROUND
#undef R2

		state_ab = vaddq_u64(save_ab, ab);
		state_cd = vaddq_u64(save_cd, cd);
		state_ef = vaddq_u64(save_ef, ef);
		state_gh = vaddq_u64(save_gh, gh);

		p += 128;
	}

	vst1q_u64(&digest[0], state_ab);
	vst1q_u64(&digest[2], state_cd);
	vst1q_u64(&digest[4], state_ef);
	vst1q_u64(&digest[6], state_gh);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#endif