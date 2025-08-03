# Malware and Vulnerability Analysis using Graph-Synchronized Language Model

This repository contains the implementation for the paper titled _Malware and Vulnerability Analysis using Graph-Synchronized Language Model_ that got accepted into the 28th International Symposium on Research in Attacks, Intrusions, and Defenses [RAID 2025](https://raid2025.github.io/call.html), Queensland, Australia (October 2025).

## Dataset
Our dataset includes 17 libraries: 

  - OpenSSL 
  - ImageMagick 
  - Libxml2 
  - Binutils
  - Redis
  - SQLite
  - Curl 
  - Musl
  - Libmicrohttpd
  - LibTomCrypt
  - Coreutils
  - Alsa
  - Libmongoc
  - Dbus
  - Allegro
  - Igraph
  - Gsl 

We filtered the functions whose CFG had less than five basic blocks. This resulted in 43,048 unique C source code functions, which we used for our analysis.
