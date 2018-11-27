def remove_zero_terms_from_fermion_operator(fermion):
    """Remove terms that contains "n n" or "n^ n^"."""
    keys = list(fermion.terms)
    for k in keys:
        if len(k) == 0:
            continue
        before = k[0]
        for t in k[1:]:
            if before == t:
                del fermion.terms[k]
                break
            before = t
