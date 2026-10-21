


    // !!! This file is generated using emlearn !!!

    #include <stdint.h>
    

static inline int32_t fall_clf_tree_0(const int16_t *features, int32_t features_length) {
          if (features[2] < 1) {
              if (features[3] < 0) {
                  return 1;
              } else {
                  return 0;
              }
          } else {
              if (features[2] < 2) {
                  return 4;
              } else {
                  if (features[19] < 0) {
                      if (features[22] < 2) {
                          return 2;
                      } else {
                          return 3;
                      }
                  } else {
                      return 5;
                  }
              }
          }
        }
        

static inline int32_t fall_clf_tree_1(const int16_t *features, int32_t features_length) {
          if (features[15] < 1) {
              if (features[21] < 7) {
                  if (features[5] < 0) {
                      return 1;
                  } else {
                      if (features[21] < 4) {
                          return 5;
                      } else {
                          return 3;
                      }
                  }
              } else {
                  if (features[2] < 2) {
                      return 4;
                  } else {
                      return 2;
                  }
              }
          } else {
              return 0;
          }
        }
        

static inline int32_t fall_clf_tree_2(const int16_t *features, int32_t features_length) {
          if (features[22] < 3) {
              if (features[7] < 0) {
                  return 2;
              } else {
                  if (features[5] < 0) {
                      return 1;
                  } else {
                      if (features[1] < 0) {
                          if (features[9] < 0) {
                              return 5;
                          } else {
                              return 4;
                          }
                      } else {
                          return 0;
                      }
                  }
              }
          } else {
              return 3;
          }
        }
        

static inline int32_t fall_clf_tree_3(const int16_t *features, int32_t features_length) {
          if (features[5] < 0) {
              if (features[21] < 7) {
                  if (features[10] < 1) {
                      return 0;
                  } else {
                      if (features[22] < 2) {
                          return 1;
                      } else {
                          return 3;
                      }
                  }
              } else {
                  if (features[12] < 0) {
                      return 2;
                  } else {
                      return 4;
                  }
              }
          } else {
              return 5;
          }
        }
        

static inline int32_t fall_clf_tree_4(const int16_t *features, int32_t features_length) {
          if (features[20] < 9) {
              if (features[8] < 0) {
                  if (features[14] < 3) {
                      if (features[14] < 2) {
                          return 3;
                      } else {
                          return 2;
                      }
                  } else {
                      if (features[0] < 5) {
                          return 4;
                      } else {
                          return 5;
                      }
                  }
              } else {
                  return 1;
              }
          } else {
              return 0;
          }
        }
        

static inline int32_t fall_clf_tree_5(const int16_t *features, int32_t features_length) {
          if (features[21] < 4) {
              return 5;
          } else {
              if (features[1] < 0) {
                  if (features[11] < 0) {
                      if (features[22] < 2) {
                          return 2;
                      } else {
                          if (features[0] < 1) {
                              return 3;
                          } else {
                              return 4;
                          }
                      }
                  } else {
                      return 1;
                  }
              } else {
                  return 0;
              }
          }
        }
        

static inline int32_t fall_clf_tree_6(const int16_t *features, int32_t features_length) {
          if (features[0] < 10) {
              if (features[5] < 0) {
                  return 1;
              } else {
                  if (features[23] < 4) {
                      return 2;
                  } else {
                      if (features[0] < 5) {
                          if (features[14] < 3) {
                              return 3;
                          } else {
                              return 4;
                          }
                      } else {
                          return 5;
                      }
                  }
              }
          } else {
              return 0;
          }
        }
        

static inline int32_t fall_clf_tree_7(const int16_t *features, int32_t features_length) {
          if (features[13] < 1) {
              if (features[0] < 1) {
                  return 3;
              } else {
                  if (features[15] < 0) {
                      if (features[0] < 1) {
                          return 2;
                      } else {
                          return 4;
                      }
                  } else {
                      if (features[16] < 2) {
                          return 1;
                      } else {
                          return 5;
                      }
                  }
              }
          } else {
              return 0;
          }
        }
        

static inline int32_t fall_clf_tree_8(const int16_t *features, int32_t features_length) {
          if (features[21] < 7) {
              if (features[12] < 0) {
                  return 3;
              } else {
                  if (features[4] < 2) {
                      if (features[13] < 1) {
                          return 1;
                      } else {
                          return 0;
                      }
                  } else {
                      return 5;
                  }
              }
          } else {
              if (features[11] < 0) {
                  return 2;
              } else {
                  return 4;
              }
          }
        }
        

static inline int32_t fall_clf_tree_9(const int16_t *features, int32_t features_length) {
          if (features[4] < 2) {
              if (features[4] < 2) {
                  return 0;
              } else {
                  return 1;
              }
          } else {
              if (features[22] < 2) {
                  return 2;
              } else {
                  if (features[21] < 4) {
                      return 5;
                  } else {
                      if (features[12] < 0) {
                          return 3;
                      } else {
                          return 4;
                      }
                  }
              }
          }
        }
        

int32_t fall_clf_predict(const int16_t *features, int32_t features_length) {

        int32_t votes[6] = {0,};
        int32_t _class = -1;

        _class = fall_clf_tree_0(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_1(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_2(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_3(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_4(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_5(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_6(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_7(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_8(features, features_length); votes[_class] += 1;
    _class = fall_clf_tree_9(features, features_length); votes[_class] += 1;
    
        int32_t most_voted_class = -1;
        int32_t most_voted_votes = 0;
        for (int32_t i=0; i<6; i++) {

            if (votes[i] > most_voted_votes) {
                most_voted_class = i;
                most_voted_votes = votes[i];
            }
        }
        return most_voted_class;
    }
    

int fall_clf_predict_proba(const int16_t *features, int32_t features_length, float *out, int out_length) {

        int32_t _class = -1;

        for (int i=0; i<out_length; i++) {
            out[i] = 0.0f;
        }

        _class = fall_clf_tree_0(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_1(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_2(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_3(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_4(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_5(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_6(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_7(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_8(features, features_length); out[_class] += 1.0f;
    _class = fall_clf_tree_9(features, features_length); out[_class] += 1.0f;
    
        // compute mean
        for (int i=0; i<out_length; i++) {
            out[i] = out[i] / 10;
        }
        return 0;
    }
    