
import tensorflow as tf

def ola():
    C = tf.cast(tf.floor(C, name='coord_int'),tf.int32) #---->[B,i*N,2]
    C = tf.floor(C, name='coord_int')  # ---->[B,i*N,2]

    graph = tf.get_default_graph()
    with graph.gradient_override_map({"cast": "Identity"}):
        C_out = tf.cast(C, tf.int32)  # ---->[B,i*N,2]

    # Make indices and updates for scatter
    # Make batch indices
    batch_idx = tf.range(0, self.batch_size,dtype=tf.int32,name='batch_idx')
    batch_idx = tf.reshape(batch_idx, (self.batch_size, 1, 1),name='batch_idx_reshape')
    b = tf.tile(batch_idx, (1,self.N*self.i,1), name='indice_b')
    # Make channel indices
    channel = tf.zeros(shape=[self.batch_size,self.N*self.i,1],dtype=tf.int32, name='indice_channels')
    # Group indices (b,W,H,c)
    indices = tf.concat((b,C_out,channel),axis=2, name='indices')  #---->[B,i*N,4] of (b,W,H,c)
    indices = tf.reshape(tensor=indices, shape=[self.batch_size*self.i*self.N,4], name='only_indices')
    # Make updates
    updates = tf.ones(shape=[self.batch_size*self.i*self.N], dtype=tf.float32, name='updates') #---->[B*i*N] values per indice
    # Build pixel space with saturation ( caused by sum of repeated indices)
    pixel_space_act = tf.scatter_nd(indices=indices,
                        updates=updates,
                        shape=[self.batch_size, self.output_width, self.output_height, 1], name='pixel_space_act')

    # Remove saturation
    #pixel_space_act = tf.clip_by_value(pixel_space_act, 0.0, 1.0, name='pixel_space_act_clip')#--> color [0.0 or 1.0]

    #return pixel_space_act
    return C_out