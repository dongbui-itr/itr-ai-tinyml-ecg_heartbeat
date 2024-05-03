import numpy as np

label_to_number = {
    'N': 0,
    'V': 1,
    'S': 2,
    'I': 3,
    'Q': 4
}


def bxb(predict_sample, predict_symbol, ref_sample, ref_symbol, sampling_rate=250):
    epsilon = 0.15 * sampling_rate

    # Find true sample
    matrix = ref_sample[:, None] - predict_sample
    matrix = np.abs(matrix)
    compare_matrix = np.less_equal(matrix, epsilon)
    true_sample = np.sum(compare_matrix, axis=-1)

    # Find true symbols of true sample
    true_sample_index = np.nonzero(compare_matrix)
    compare_symbol = np.array(['+'] * len(ref_symbol))
    compare_symbol[true_sample_index[0]] = predict_symbol[true_sample_index[1]]
    true_symbol = ref_symbol == compare_symbol

    # Convert symbols to numbers
    ref_symbol_number = np.array([label_to_number[i] for i in ref_symbol])

    return np.concatenate([[ref_sample], [true_sample]]), np.concatenate([[ref_symbol_number], [true_symbol]])


def test():
    sample_ref = np.array([250, 500, 1000, 2000, 2500, 3333])
    symbol_ref = np.array(['N', 'S', 'V', 'N', 'I', 'Q'])

    sample_pred = np.array([250, 550, 750, 1100, 1234, 2002, 2550, 3333, 5555])
    symbol_pred = np.array(['N', 'N', 'S', 'V', 'V', 'N', 'Q', 'I', 'Q'])

    a, b = bxb(sample_pred, symbol_pred, sample_ref, symbol_ref)

    print(a)
    print(b)


if __name__ == '__main__':
    test()
