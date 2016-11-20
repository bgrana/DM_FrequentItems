import argparse

from apriori import Frequent_items


def main(args):
    # Load transactions
    fi = Frequent_items(args["in"])
    # Compute support threshold as a percentage of the 
    # number of transactions
    sth = round(fi.ntrans*args["s"])
    
    # Find frequent itemsets
    print("\nLooking for frequent items with apriori algorithm...")
    freq = fi.get_frequent_items(sth)
    if freq:
        print("The itemsets returned by apriori algorithm with support threshold " + str(args["s"]*100) + "% are:")
        print(freq)
    else:
        print("No frequent items found for support threshold " + str(args["s"]*100) + "%.")

    print()

    # Find association rules based on common itemsets
    print("Looking for association rules...")
    rules = fi.get_rules(args["c"], freq)
    if rules:
        print("The found association rules with confidence " + str(args["c"]*100) + "% are:")
        fi.print_rules(rules)
    else:
        print("No association rules found for confidence " + str(args["c"]*100) + "%.")


if  __name__ =='__main__':
    # Parse input
    parser = argparse.ArgumentParser(description='Find similar documents using shingling, minhashing and LSH.')
    parser.add_argument('--in', required=True, type=str, help='Path to the input file containing transactions (one per row).')
    parser.add_argument('--s', default=0.05, type=float, help='Support threshold as a percentage on the number of transactions (from 0 to 1). Defaults to 0.05.')
    parser.add_argument('--c', default=0.5, type=float, help='Confidence threshold (from 0 to 1). Defaults to 0.5.')

    args = vars(parser.parse_args())
    main(args)