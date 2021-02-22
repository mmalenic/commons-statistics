import os

import numpy as np
from scipy import stats


def generate_truncnorm_test_values():
    # Defaults
    print_test_values((1.9, 1.3, -1.1, 3.4),
                      [0, 0.0001, 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.900, 0.950, 0.975, 0.990,
                       0.999, 0.9999, 1])
    # one-sided lower tail
    print_test_values((12, 2.4, -np.inf, 7.1))
    # one-sided upper tail
    print_test_values((-9.6, 17, -15, np.inf))
    # no truncation
    print_test_values((3, 1.1, -np.inf, np.inf))
    # lower tail only
    print_test_values((0, 1, -np.inf, -5))
    # upper tail only
    print_test_values((0, 1, 5, np.inf))
    # narrow truncated range
    print_test_values((7.1, 9.9, 7.0999999, 7.1000001))


def code_format_values(input_parameters, percentiles=None):
    ppf_values, cdf_values, pdf_values, mean, var = create_test_values(input_parameters, percentiles)
    out = "new TruncatedNormalDistribution({:.15g}, {:.15g}, {:.15g}, {:.15g})".format(*input_parameters) + "," + \
          os.linesep + "new double[] " + format_values(ppf_values) + "," + os.linesep + "new double[] " + \
          format_values(cdf_values) + "," + os.linesep + "new double[] " + format_values(pdf_values) + "," + \
          os.linesep + "{:.15g},".format(mean) + os.linesep + "{:.15g}".format(var) + os.linesep
    out = out.replace("-inf", "Double.NEGATIVE_INFINITY")
    out = out.replace("inf", "Double.POSITIVE_INFINITY")
    return out


def print_test_values(input_parameters, percentiles=None):
    ppf_values, cdf_values, pdf_values, mean, var = create_test_values(input_parameters, percentiles)
    print("for mean = {:.15g}, std = {:.15g}, lower = {:.15g}, upper = {:.15g}:".format(*input_parameters))
    print("ppf values: ", format_values(ppf_values))
    print("cdf values: ", format_values(cdf_values))
    print("pdf values: ", format_values(pdf_values))
    print("mean = {:.15g}".format(mean))
    print("variance = {:.15g}".format(var))
    print()
    print(code_format_values(input_parameters, percentiles))


def create_test_values(input_parameters, percentiles=None):
    truncnorm = stats.truncnorm(*bound_for(*input_parameters))

    if percentiles is None:
        mean = truncnorm.mean()
        std = truncnorm.std()
        cdf_points = [mean - 5 * std, mean - 4 * std, mean - 3 * std, mean - 2 * std, mean - 1 * std, mean - std, mean,
                      mean + std, mean + 2 * std, mean + 3 * std, mean + 4 * std, mean + 5 * std]
        ppf_points = [0] + [truncnorm.cdf(x) for x in cdf_points] + [1]
        percentiles = list(dict.fromkeys(ppf_points))

    cdf_values = truncnorm.ppf(percentiles)
    pdf_values = truncnorm.pdf(cdf_values)
    return percentiles, cdf_values, pdf_values, truncnorm.mean(), truncnorm.var()


def bound_for(mean, sd, a, b):
    return (a - mean) / sd, (b - mean) / sd, mean, sd


def format_values(values):
    return '{ ' + ', '.join('{:.15g}'.format(x) for x in values) + ' }'


if __name__ == '__main__':
    generate_truncnorm_test_values()
