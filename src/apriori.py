from collections import deque
from itertools import combinations, permutations

import numpy as np


class Frequent_items:
    """Implementation of the Apriori algorithm for frequent itemsets detection proposed
    in the paper 'Fast Algorithms for Mining Association Rules' by R. Agrawal and R. Srikant.
    """
    def __init__(self, filepath):
        """
        Initialization method that receives a filepath from which read information.
        
        filepath:   Path to the file containing the transactions (one per line),
                    each represented as a set of integers separated by spaces.
        """
        with open(filepath, "r") as f:
            self.items = {}
            self.ntrans = 0
            for i,line in enumerate(f.readlines()):
                for elem in line.split():
                    try:
                        elem = int(elem)
                    except ValueError:
                        pass
                    if elem not in self.items:
                        self.items[elem] = set()
                    self.items[elem].add(i)
                self.ntrans += 1
            self.c1 = [({elem}, len(indices)) for elem, indices in self.items.items()]

    def _get_support(self, itemsets):
        """
        This method computes the support of a collection of itemsets based on the
        transactions.
        
        itemsets: An iterable of Python sets representing itemsets.
        returns:  An iterable of 2-tuples where the first element is
                  a set (representing an itemset) and the second one
                  is an integer (representing the support).
        """
        # Initialize counts
        supports = deque()
        for iset in itemsets:
            common_indices = set.intersection(*[self.items[item] for item in iset])
            supports.append((iset, len(common_indices)))
        return supports
    
    def _next_candidates(self, lprev):
        """
        Find the set of candidates based on the previous frequent
        (k-1)-itemsets.
        
        lprev:      The collection of previous large (k-1)-itemsets as
                    an an iterable of sets.
        returns:    An iterable of 2-tuples where the first element is
                    a set (representing an itemset) and the second one
                    is an integer (representing the support).
                
        """
        lprev = [set(itemset) for itemset in lprev.keys()]
        k = len(lprev[0])+1
        
        # Join (k-1)-itemsets to get all candidates of size k
        allcandidates = [s1 | s2 for s1 in lprev for s2 in lprev
                                 if len(s1 | s2) == k]
        
        # Filter out candidates which have some (k-1) subset not
        # identified as a large (k-1)-itemset
        candidates = deque()
        for iset in allcandidates:
            # Compute subsets of k-1 elements
            # Number of (k-1) combinations for a set of
            # k elements is precisely k (binomial coefficient(k,k-1))
            # so len(subsets) is k
            subsets = [set(x) for x in combinations(iset, k-1)]
            for i in range(k):
                if subsets[i] not in lprev:
                    break
                if i == k-1 and iset not in candidates: # Last iteration
                    candidates.append(iset)
            
        # Return candidates with their corresponding support
        return self._get_support(candidates)

    def _filter_candidates(self, candidates):
        """
        This methods select only those candidates such that their
        support is greater than or equal the support threshold.
        
        candidates: An iterable of 2-tuples where the first element is
                    a set (representing an itemset) and the second one
                    is an integer (representing the support).
        returns:    An iterable of itemsets as sets.
        """
        # Filter out itemsets with low support
        return {tuple(itemset):sup for itemset,sup in candidates
                        if sup >= self.minsup}
    
    def get_frequent_items(self, minsup):
        """
        Get the frequent items of the loaded transactions based on the
        provided support threshold.
        
        minsup:    Support threshold for the itemsets filtering.
        returns:   The identified frequent items as a set of tuples.
        """
        # Initialize variables
        self.minsup = minsup
        l = self._filter_candidates(self.c1)
        answer = l
        
        # Updates candidates and answer
        while l:
            ck = self._next_candidates(l)
            l = self._filter_candidates(ck)
            answer.update(l)
        
        return answer
    
    def _get_confidence(self, union_supp, pre_supp):
        """
        Compute confidence based on itemsets supports
        """
        return union_supp/pre_supp
    
    def get_rules(self, min_confidence, itemsets):
        """
        Get association rules based on the provided
        confidence threshold.
        """
        relevant_items = {}
        for item,support in itemsets.items():
            if len(item) > 1:
                
                # Add >2-tuple value
                relevant_items[frozenset(item)] = support
                
                # Add 1-tuple values
                relevant_items[frozenset([item[0]])] = itemsets[tuple([item[0]])]
                relevant_items[frozenset([item[1]])] = itemsets[tuple([item[1]])]
        
        possible_rules = permutations(relevant_items, 2)
        rules = []
        for rule in possible_rules:
            if set(rule[0]) & set(rule[1]) or frozenset(set(rule[0]) | set(rule[1])) not in relevant_items.keys():
                continue
            
            union_supp = relevant_items[frozenset(set(rule[0]) | set(rule[1]))]
            pre_supp = relevant_items[frozenset(rule[0])]

            if self._get_confidence(union_supp, pre_supp) >= min_confidence:
                rules.append((tuple(rule[0]),tuple(rule[1])))

        return rules

    def print_rules(self, rules):
        """
        Print the association rules returned by get_rules in
        a more readable way.
        """
        [print(str(set(rule[0])) + " => " + str(set(rule[1]))) for rule in rules]
