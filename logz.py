import os
import os.path as osp
import time
import atexit


class G:
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}


def configure_output_dir(d=None):
    G.first_row = True
    G.log_headers = []
    G.log_current_row = {}
    G.output_dir = d or "/tmp/experiments/%i" % int(time.time())
    if not osp.exists(G.output_dir):
        os.makedirs(G.output_dir)
    G.output_file = open(osp.join(G.output_dir, "log.txt"), 'w')
    atexit.register(G.output_file.close)


def log_tabular(key, val):
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
    assert key not in G.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
    G.log_current_row[key] = val


def dump_tabular():
    vals = []
    key_lens = [len(key) for key in G.log_headers]
    max_key_len = max(15, max(key_lens))
    keystr = '%' + '%d' % max_key_len
    fmt = "| " + keystr + "s | %15s |"
    n_slashes = 22 + max_key_len
    print("-" * n_slashes)
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        if hasattr(val, "__float__"):
            valstr = "%8.3g" % val
        else:
            valstr = val
        print(fmt % (key, valstr))
        vals.append(val)
    print("-" * n_slashes)
    if G.output_file is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
        G.output_file.write("\t".join(map(str, vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()
    G.first_row = False