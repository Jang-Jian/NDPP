include make/make.configure.mk


ifeq ($(WITH_PYTHON), 1)

# How to silence internal boost deprecated messages: 
# https://stackoverflow.com/questions/66906613/how-to-silence-internal-boost-deprecated-messages
SILENCE_BOOST_WARN=-DBOOST_ALLOW_DEPRECATED_HEADERS -DBOOST_BIND_GLOBAL_PLACEHOLDERS

CC_DEPENDENCY+=$(SILENCE_BOOST_WARN) -I$(PYTHON_INCLUDE) -lpython$(PY3_VER) -I$(BOOST_INCLUDE) -L$(BOOST_LIBS) -lboost_python$(PY3_BOOST) -lboost_numpy$(PY3_BOOST)

endif