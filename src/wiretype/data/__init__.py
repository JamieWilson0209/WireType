"""Connectome intake: edge lists, annotations, and the graphs built from them.

FAFB v783 (FlyWire) is the pretraining volume; MCNS v1.0 is the transfer
volume. Both arrive here as a node table and a weighted directed edge list,
so that everything downstream sees one shape regardless of source.
"""
