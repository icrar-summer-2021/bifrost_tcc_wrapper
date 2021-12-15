inc_dir = 'group/director2183/dancpr/src/bifrost_tcc_wrapper/src'
pybin = '/pawsey/centos7.6/apps/gcc/4.8.5/python/3.6.3/bin/python'
plugin = 'btcc'



def generate_and_patch(plugin, inc_dir=None, pybin='python'):
    cmd = f"""
{pybin} -c 'from ctypesgen import main as ctypeswrap; ctypeswrap.main()' -l{plugin} -I. -I{inc_dir} {plugin}.h -o {plugin}_generated.py
# WAR for 'const char**' being generated as POINTER(POINTER(c_char)) instead of POINTER(c_char_p)
sed -i 's/POINTER(c_char)/c_char_p/g' {plugin}_generated.py
# WAR for a buggy WAR in ctypesgen that breaks type checking and auto-byref functionality
sed -i 's/def POINTER/def POINTER_not_used/' {plugin}_generated.py
# WAR for a buggy WAR in ctypesgen that breaks string buffer arguments (e.g., as in address.py)
sed -i 's/class String/String = c_char_p\\nclass String_not_used/' {plugin}_generated.py
sed -i 's/String.from_param/String_not_used.from_param/g' {plugin}_generated.py
sed -i 's/def ReturnString/ReturnString = c_char_p\\ndef ReturnString_not_used/' {plugin}_generated.py
sed -i '/errcheck = ReturnString/s/^/#/' {plugin}_generated.py"""

    import os
    for line in cmd.split('\n'):
        print(line)
        os.system(line)
        print('\n')

generate_and_patch(plugin, inc_dir, pybin)