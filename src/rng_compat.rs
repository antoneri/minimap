use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub trait TrialRng {
    fn randomized_index_vector(&mut self, out: &mut [u32]);
}

#[derive(Debug, Clone)]
pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut rng = Self {
            mt: [0u32; 624],
            idx: 624,
        };
        rng.seed(seed);
        rng
    }

    pub fn seed(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..624 {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.idx = 624;
    }

    fn twist(&mut self) {
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;

        for i in 0..624 {
            let x = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % 624] & LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MATRIX_A;
            }
            self.mt[i] = self.mt[(i + 397) % 624] ^ x_a;
        }
        self.idx = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn rand_int_inclusive(&mut self, min: u32, max: u32) -> u32 {
        if min >= max {
            return min;
        }

        // Match libc++ std::uniform_int_distribution<unsigned int> behavior with
        // mt19937: build random values from the low w bits and reject above range.
        let range = max.wrapping_sub(min).wrapping_add(1);
        if range == 1 {
            return min;
        }
        if range == 0 {
            return self.next_u32();
        }

        let mut w = 32usize - range.leading_zeros() as usize - 1;
        let low_mask = if w == 0 { 0 } else { u32::MAX >> (32 - w) };
        if (range & low_mask) != 0 {
            w += 1;
        }
        let mask = if w == 0 { 0 } else { u32::MAX >> (32 - w) };

        loop {
            let u = self.next_u32() & mask;
            if u < range {
                return min + u;
            }
        }
    }

    pub fn randomized_index_vector(&mut self, out: &mut [u32]) {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = i as u32;
        }
        let size = out.len();
        for i in 0..size {
            let j = i + self.rand_int_inclusive(0, (size - i - 1) as u32) as usize;
            out.swap(i, j);
        }
    }
}

impl TrialRng for Mt19937 {
    fn randomized_index_vector(&mut self, out: &mut [u32]) {
        Mt19937::randomized_index_vector(self, out);
    }
}

#[derive(Debug, Clone)]
pub struct RustRng {
    rng: SmallRng,
}

impl RustRng {
    pub fn new(seed: u32) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed as u64),
        }
    }
}

impl TrialRng for RustRng {
    fn randomized_index_vector(&mut self, out: &mut [u32]) {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = i as u32;
        }
        let size = out.len();
        for i in 0..size {
            let j = self.rng.gen_range(i..size);
            out.swap(i, j);
        }
    }
}
