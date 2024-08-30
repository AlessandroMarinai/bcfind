def equalize_lists_length(list1, list2, ground_truth, duplicate=False):
    # Calcola la lunghezza massima tra list1, list2 e ground_truth
    if duplicate:
        list1 = list1 * 2
        ground_truth = ground_truth * 2
        list2 = list2 * 2
    
    max_len = max(len(list1), len(list2), len(ground_truth))

    # Verifica se list2 e ground_truth hanno lunghezze diverse
    if len(list2) != len(ground_truth):
        raise ValueError(" e ground_truth devono avere la stessa lunghezza")

    # Estende list1, list2 e ground_truth fino alla lunghezza massima
    list1_extension = list1 * (max_len // len(list1)) + list1[:max_len % len(list1)]
    list2_extension = list2 * (max_len // len(list2)) + list2[:max_len % len(list2)]
    ground_truth_extension = ground_truth * (max_len // len(ground_truth)) + ground_truth[:max_len % len(ground_truth)]

    return list1_extension, list2_extension, ground_truth_extension
# Esempio d'uso senza raddoppio:
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6]
ground_truth = ['a', 'b', 'c']

list1, list2, ground_truth = equalize_lists_length(list1, list2, ground_truth)
print("Lista 1 senza raddoppio:", list1)
print("Lista 2 senza raddoppio:", list2)
print("Ground truth senza raddoppio:", ground_truth)

# Esempio d'uso con raddoppio:
list1 = [1, 2, 3, 4 ,5]
list2 = [4, 5, 6]
ground_truth = ['a', 'b', 'c']

list1, list2, ground_truth = equalize_lists_length(list1, list2, ground_truth, duplicate=True)
print("Lista 1 con raddoppio:", list1)
print("Lista 2 con raddoppio:", list2)
print("Ground truth con raddoppio:", ground_truth)
