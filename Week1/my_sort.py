# my_sort.py


# Sort using Python's built-in sorting function
sorted_list1 = sorted(my_list)


def custom_sort(array):
    n = len(array)
    sorted_array = array.copy()
    for i in range(n):
        for j in range(0, n-i-1):
            if sorted_array[j] > sorted_array[j+1]:
                sorted_array[j], sorted_array[j+1] = sorted_array[j+1], sorted_array[j]
    return sorted_array


sorted_list2 = custom_sort(my_list)
