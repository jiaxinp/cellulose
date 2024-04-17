def custom_sort(s):
    # Split the string into alphabetic part and numerical part
    alpha_part = ''.join(filter(str.isalpha, s))
    num_part = int(''.join(filter(str.isdigit, s)))
    return alpha_part, num_part