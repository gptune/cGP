import runpy
thisdict = {
  "RND_SEED": 123
}
file_path = 'cGP_mod.py'
RND_SEED = 123
runpy.run_path(file_path, init_globals=thisdict)
