import numpy as np


class SVDProcessor:

    def __init__(self):
        pass

    # -------------------------------------------------
    # Validate
    # -------------------------------------------------

    def validate(self, matrix):

        if matrix is None:
            raise ValueError("Matrix is None.")

        if matrix.shape != (8, 8):
            raise ValueError("Expected 8x8 DCT block.")

    # -------------------------------------------------
    # Forward SVD
    # -------------------------------------------------

    def forward(self, matrix):

        self.validate(matrix)

        U, S, VT = np.linalg.svd(
            matrix,
            full_matrices=True
        )

        return U, S, VT

    # -------------------------------------------------
    # Inverse SVD
    # -------------------------------------------------

    def inverse(self, U, S, VT):

        sigma = np.zeros((8, 8))

        np.fill_diagonal(
            sigma,
            S
        )

        return U @ sigma @ VT

    # -------------------------------------------------
    # Sigma Matrix
    # -------------------------------------------------

    def sigma_matrix(self, S):

        sigma = np.zeros((8, 8))

        np.fill_diagonal(
            sigma,
            S
        )

        return sigma
    
        # -------------------------------------------------
    # Largest Singular Value
    # -------------------------------------------------

    def largest_sigma(self, S):

        return S[0]

    # -------------------------------------------------
    # Smallest Singular Value
    # -------------------------------------------------

    def smallest_sigma(self, S):

        return S[-1]

    # -------------------------------------------------
    # Sigma Statistics
    # -------------------------------------------------

    def statistics(self, S):

        return {

            "max": float(np.max(S)),

            "min": float(np.min(S)),

            "mean": float(np.mean(S)),

            "std": float(np.std(S))
        }

    # -------------------------------------------------
    # Energy
    # -------------------------------------------------

    def energy(self, S):

        return np.sum(
            np.square(S)
        )
    
        # -------------------------------------------------
    # Embed Single Bit
    # -------------------------------------------------

    def embed_bit(
        self,
        S,
        bit,
        alpha=0.05
    ):

        S = S.copy()

        if bit == 1:

            S[0] += alpha

        else:

            S[0] -= alpha

        return S

    # -------------------------------------------------
    # Extract Single Bit
    # -------------------------------------------------

    def extract_bit(
        self,
        original_sigma,
        modified_sigma
    ):

        if modified_sigma[0] > original_sigma[0]:

            return 1

        return 0
    
        # -------------------------------------------------
    # Embed Multiple Bits
    # -------------------------------------------------

    def embed_bits(
        self,
        sigma,
        bits,
        alpha=0.05
    ):

        sigma = sigma.copy()

        count = min(
            len(bits),
            len(sigma)
        )

        for i in range(count):

            if bits[i]:

                sigma[i] += alpha

            else:

                sigma[i] -= alpha

        return sigma

    # -------------------------------------------------
    # Extract Multiple Bits
    # -------------------------------------------------

    def extract_bits(
        self,
        original,
        modified
    ):

        bits = []

        for o, m in zip(original, modified):

            if m > o:

                bits.append(1)

            else:

                bits.append(0)

        return bits