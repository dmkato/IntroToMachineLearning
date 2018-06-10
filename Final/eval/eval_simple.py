import os.path
import sys, getopt
import csv
import numpy as np
import sys, time
#-- HAMED SHAHBAZI

def read_and_check(pred, gold):
    #reading pred and gold and filling gps_data

    #check if files exist
    for f in [pred, gold]:
        if not os.path.exists(f):
            print('FILE NOT FOUND: %s'%f)
            raise 'file not exist'

    #reading and checking pred and gold rows
    gps_data = []
    g_ = []
    p_ = []
    with open(gold, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines())]
        for doi, do in enumerate(data):
            if len(do) != 1:
                print('Error in:%s'% gold)
                raise 'each row in gold must contain only one value; label'
            g_.append((str(doi), int(do[0].strip())))
        f.close()
    
    with open(pred, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines())]
        for doi, do in enumerate(data):
            if len(do) != 2:
                print('Error in:%s'% pred)
                raise 'each row in pred file must contain only two values: score label'
            p_.append((str(doi), float(do[0].strip()), int(do[1].strip())))
        f.close()

    if len(g_) != len(p_):
        print('Error in :(%s, %s)'%(gold, pred))
        raise 'the number of samples in pred should be the same as gold'

    for i in xrange(len(g_)):
        if g_[i][0] != p_[i][0]:
            print('Error in (%s, %s) for sample %s'%(gold, pred, str(i)))
            raise 'sample of gold are missing in the predict'

    gps_data.append(('simple', gold, pred, g_, p_))

    return gps_data

def get_f1(g_, p_):
    #this function computes teh area under Accuracy, Precision, Recall and F1
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0

    for i in xrange(len(g_)):
        g_label = g_[i][-1]
        p_label = p_[i][-1]
        if p_label == 1:
            if g_label == 1:
                tp += 1
            else:
                fp += 1
        else:
            if g_label == 1:
                fn += 1
            else:
                tn += 1

    if (tp + fp) == 0:
        p = 1.0
    else:
        p = round(tp / (tp + fp), 3)

    if (tp + fn) == 0:
        r = 1.0
    else:
        r = round(tp / (tp + fn), 3)

    if (r + p) == 0.0:
        f1 = 0.0
    else:
        f1 = round(2 * (r * p) / (r + p), 3)

    if (tp + fp + fn + tn) == 0:
        acc = 0.0
    else:
        acc = round((tp + tn) / (tp + fp + fn + tn), 3)
    
    return acc, p, r, f1
    
def get_auc(g_, p_):
    #this function computes teh area under ROC
    gold_negs = []
    gold_pos = []
    pred_scores = {}
    np.random.seed(seed=12345)
    for i in xrange(len(g_)):
        g_label = g_[i][-1]
        pred_scores[i] = p_[i][1]
        if g_label == 1:
            gold_pos.append(i)
        else:
            gold_negs.append(i)


    auc_samples = 50000
    pos_samples = np.random.randint(0, len(gold_pos), auc_samples)
    neg_samples = np.random.randint(0, len(gold_negs), auc_samples)
    auc = round(sum([1.0 for x, y in zip(pos_samples, neg_samples)\
     if pred_scores[gold_pos[x]] > pred_scores[gold_negs[y]]]) / auc_samples, 3)
    return auc


def evaluate(gp_data):
    #this function computes evaluations for one submission
    set_id, gold, pred, g_, p_ = gp_data
    
    #print('\nset: %s'%set_id)
    a, p, r, f1 = get_f1(g_, p_)
    auc = get_auc(g_, p_)
    print('Precision: %s \t Recall: %s \t F1: %s \t AUC: %s'%\
        (str(round(p, 3)), str(round(r, 3)), str(round(f1, 3)), str(round(auc, 3))))
    return a, p, r, f1, auc

def main(argv):

    pred = 'pred.csv'
    gold = 'gold.csv'
    result = 'result.csv'
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    try:
       opts, args = getopt.getopt(argv,"h:p:g:o:",["ifile=","ofile="])
    except getopt.GetoptError:
       print 'eval_simple.py -p pred.csv -g gold.csv -o result.csv'
       sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'eval_simple.py -p pred.csv -g gold.csv -o result.csv'
            sys.exit()
        elif opt in ("-p", "--ifile"):
            pred = arg
        elif opt in ("-g", "--ofile"):
            gold = arg
        elif opt in ("-o", "--ofile"):
            result = arg

    print('_'*50)
    pred, gold = [__location__ + '/' + pred, __location__ + '/' + gold]
    print(pred)
    print(gold)
    outs = []
    gps_data = read_and_check(pred, gold)
    for gp_data in gps_data:
        a, p, r, f1, auc = evaluate(gp_data)
        outs.append(['running', gp_data[0], str(a), str(p), str(r), str(f1), str(auc)])

    out_f = __location__ + '/' + result
    out_w = open(out_f, 'w')

    out_w.write('{0[0]:<25}{0[1]:<30}{0[2]:<10}{0[3]:<10}{0[4]:>10}{0[5]:>10}{0[6]:>10}'.format(['sub', 'set', \
            'A', 'P', 'R', 'F1', 'AUC']) + '\n')
    
    for out in outs:
        out_w.write('{0[0]:<25}{0[1]:<30}{0[2]:<10}{0[3]:<10}{0[4]:>10}{0[5]:>10}{0[6]:>10}'.format(out) + '\n')

    out_w.flush()
    out_w.close()

    print('_'*50)



if __name__ == "__main__":
    main(sys.argv[1:])


