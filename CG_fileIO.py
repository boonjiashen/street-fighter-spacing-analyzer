"""Loaders and savers for file I/O of center of gravity of Street Fighter 4
characters in a match
"""

import re

class CG_fileIO():

    def saveline(filename, fi, p1_CG, p2_CG, append=True):
        """Save a snippet of CG info into file
        
        `p1_CG` `p2_CG` = (x, y) tuple of two player's CG

        `append` if True, append rather than overwrite file
        """

        fid = open(filename, 'a' if append else 'w')
        snippet = ' '.join(map(str, [fi, p1_CG, p2_CG]))
        fid.write(snippet + '\n')
        fid.close()

    def load(filename):
        """Return a dictionary that maps the frame index of a video to a
        (p1's CG, p2's CG) 2-ple

        Essentially an inverse of the saver.
        """

        CGs = {}
        pattern = r"(.*) (None|\(.*\)) (None|\(.*\))"
        with open(filename, 'r') as fid:
            for line in fid:
                match = re.match(pattern, line)
                frame_index, p1_CG, p2_CG = map(eval, match.groups())
                CGs[frame_index] = (p1_CG, p2_CG)

        return CGs


