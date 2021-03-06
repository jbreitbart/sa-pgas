/*   $Source: /var/local/cvs/gasnet/vapi-conduit/gasnet_core_connect.c,v $
 *     $Date: 2011/05/02 22:27:46 $
 * $Revision: 1.58 $
 * Description: Connection management code
 * Copyright 2011, E. O. Lawrence Berekely National Laboratory
 * Terms of use are as specified in license.txt
 */

#include <gasnet_internal.h>
#include <gasnet_core_internal.h>

/* For open(), stat(), O_CREAT, etc.: */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

/*
  The following configuration cannot yet be overridden by environment variables.
*/
#if GASNET_CONDUIT_VAPI
  #define GASNETC_QP_PATH_MTU           MTU1024
  #define GASNETC_QP_MIN_RNR_TIMER      IB_RNR_NAK_TIMER_0_08
#else
  #define GASNETC_QP_PATH_MTU           IBV_MTU_1024
  #define GASNETC_QP_MIN_RNR_TIMER      6       /*IB_RNR_NAK_TIMER_0_08*/
#endif
#define GASNETC_QP_STATIC_RATE          0
#define GASNETC_QP_RNR_RETRY            7       /* retry forever, but almost never happens */

#define GASNETC_PSN(_sender, _qpi)  ((_sender)*gasnetc_alloc_qps + (_qpi))

/* Convenience iterators */
#define GASNETC_FOR_EACH_QPI(_conn_info, _qpi, _cep)  \
  for((_cep) = (_conn_info)->cep, (_qpi) = 0; \
      (_qpi) < gasnetc_alloc_qps; ++(_cep), ++(_qpi))

/* ------------------------------------------------------------------------------------ */
/* Global data */

gasneti_semaphore_t gasnetc_zero_sema = GASNETI_SEMAPHORE_INITIALIZER(0, 0);

int gasnetc_ud_rcvs = 0;
int gasnetc_ud_snds = 0;

/* ------------------------------------------------------------------------------------ */

/* Common types */
typedef GASNETC_IB_CHOOSE(VAPI_qp_attr_t,       struct ibv_qp_attr)     gasnetc_qp_attr_t;
typedef GASNETC_IB_CHOOSE(VAPI_qp_attr_mask_t,  enum ibv_qp_attr_mask)  gasnetc_qp_mask_t;
typedef GASNETC_IB_CHOOSE(VAPI_ud_av_t,         struct ibv_ah_attr)     gasnetc_ah_attr_t;
typedef GASNETC_IB_CHOOSE(VAPI_ud_av_hndl_t,    struct ibv_ah *)        gasnetc_ib_ah_t;

typedef struct {
  gasneti_weakatomic_t ref_count;
  gasnetc_ib_ah_t ib_ah;
} gasnetc_ah_t;

/* Info used for connection establishment */
typedef struct {
  gasnet_node_t     node;
  gasnetc_cep_t     *cep;        /* Vector of gasnet endpoints */
  gasnetc_qpn_t     *local_qpn;  /* Local qpns of connections */
  gasnetc_qpn_t     *remote_qpn; /* Remote qpns of connections */
#if GASNETC_IBV_XRC
  gasnetc_qpn_t     *local_xrc_qpn;  /* Local qpns of XRC rcv qps */
  gasnetc_qpn_t     *remote_xrc_qpn; /* Remote qpns of XRC rcv qps */
  uint32_t          *xrc_remote_srq_num; /* Remote SRQ numbers */
#endif
  const gasnetc_port_info_t **port;
} gasnetc_conn_info_t;

/* Info exchanged for connection setup */
#if GASNETC_IBV_XRC
  typedef struct {
    uint32_t        srq_num;
    gasnetc_qpn_t   xrc_qpn;
    gasnetc_qpn_t   qpn;
  } gasnetc_xrc_conn_data_t;
#else
  /* Just exchange qpn.  So no struct required */
#endif

#if GASNETC_IBV_XRC
  #define GASNETC_SND_QP_NEEDS_MODIFY(_xrc_snd_qp,_state) \
	(!gasnetc_use_xrc || ((_xrc_snd_qp).state < (_state)))
#else
  #define GASNETC_SND_QP_NEEDS_MODIFY(_xrc_snd_qp,_state) 1
#endif

static const char *gasnetc_connectfile_in  = NULL;
static const char *gasnetc_connectfile_out = NULL;

static int gasnetc_connectfile_in_base  = 10; /* Defaults to human readable/writable */
static int gasnetc_connectfile_out_base = 36; /* Defaults to most compact */

/* ------------------------------------------------------------------------------------ */

#if GASNET_DEBUG && GASNETC_DYNAMIC_CONNECT
/* Drop some UD packets randomly to aid debugging */
static unsigned int gasnetc_conn_drop_denom = 0;

#include <stdlib.h>
/* Uniformly distributed int in range [0..max_val] */
static unsigned int
gasnetc_conn_rand_int(unsigned int max_val)
{
  static unsigned short state[3];
  static int done_init = 0;

  if_pf (!done_init) {
    gasneti_tick_t now = gasneti_ticks_now();
    state[0] = getpid();
    state[1] = (unsigned short) now;
    state[2] = (unsigned short) (now >> 16);
    done_init = 1;
  }

  return (unsigned int)(((double)(max_val+1.0)) * erand48(state));
}
#endif

/* ------------------------------------------------------------------------------------ */

static gasneti_lifo_head_t sq_sema_freelist = GASNETI_LIFO_INITIALIZER;

static void
sq_sema_alloc(int count)
{
  gasneti_semaphore_t *p = (gasneti_semaphore_t *)
          gasnett_malloc_aligned(GASNETI_CACHE_LINE_BYTES, count * sizeof(gasneti_semaphore_t));
  int i;

  for (i=0; i<count; ++i, ++p) {
    gasneti_lifo_push(&sq_sema_freelist,p);
  }
}

static gasneti_semaphore_t * 
sq_sema_get(void)
{
  gasneti_semaphore_t *result;
 
  result = gasneti_lifo_pop(&sq_sema_freelist);
  if_pf (NULL == result) {
    sq_sema_alloc(1);
    result = gasneti_lifo_pop(&sq_sema_freelist);
    gasneti_assert(NULL != result);
  }

  return result;
}

/* ------------------------------------------------------------------------------------ */
static const char *
gasnetc_parse_filename(const char *filename)
{
  /* replace any '%' with node num */
  if (strchr(filename,'%')) {
    char *tmpname = gasneti_strdup(filename); /* must not modify callers string */
    char *p = strchr(tmpname,'%');
    do {
      size_t len = strlen(tmpname) + 16;
      char *buf = gasneti_malloc(len);
      *p = '\0';
      snprintf(buf,len,"%s%i%s",tmpname,(int)gasnet_mynode(),p+1);
      gasneti_free(tmpname);
      tmpname = buf;
    } while (NULL != (p = strchr(tmpname,'%')));
    filename = tmpname;
  }
  return filename;
}
/* ------------------------------------------------------------------------------------ */

#if GASNETC_IBV_XRC
typedef struct gasnetc_xrc_snd_qp_s {
  gasnetc_qp_hndl_t handle;
  enum ibv_qp_state state;
  gasneti_semaphore_t *sq_sema_p;
} gasnetc_xrc_snd_qp_t;

static gasnetc_xrc_snd_qp_t *gasnetc_xrc_snd_qp = NULL;
#define GASNETC_NODE2SND_QP(_node) \
	(&gasnetc_xrc_snd_qp[gasneti_nodeinfo[(_node)] * gasnetc_alloc_qps])

static gasnetc_qpn_t *gasnetc_xrc_rcv_qpn = NULL;

/* Create one XRC rcv QP */
static int
gasnetc_xrc_create_qp(struct ibv_xrc_domain *xrc_domain, gasnet_node_t node, int qpi) {
  const int cep_idx = node * gasnetc_alloc_qps + qpi;
  gasneti_atomic32_t *rcv_qpn_p = (gasneti_atomic32_t *)(&gasnetc_xrc_rcv_qpn[cep_idx]);
  gasnetc_qpn_t rcv_qpn = 0;
  int ret;

  gasneti_assert(gasnetc_xrc_rcv_qpn != NULL);
  gasneti_assert(sizeof(gasnetc_qpn_t) == sizeof(gasneti_atomic32_t));

  /* Create the RCV QPs once per supernode and register in the non-creating node(s) */
  /* QPN==1 is reserved, so we can use it as a "lock" value */
  if (gasneti_atomic32_compare_and_swap(rcv_qpn_p, 0, 1, 0)) {
    struct ibv_qp_init_attr init_attr;
    init_attr.xrc_domain = xrc_domain;
    ret = ibv_create_xrc_rcv_qp(&init_attr, &rcv_qpn);
    GASNETC_VAPI_CHECK(ret, "from ibv_create_xrc_rcv_qp()");

    gasneti_atomic32_set(rcv_qpn_p, rcv_qpn, 0);
  } else {
    gasneti_waituntil(1 != (rcv_qpn = gasneti_atomic32_read(rcv_qpn_p, 0)));
    ret = ibv_reg_xrc_rcv_qp(xrc_domain, rcv_qpn);
    GASNETC_VAPI_CHECK(ret, "from ibv_reg_xrc_rcv_qp()");
  }

  return GASNET_OK;
}

static int
gasnetc_xrc_modify_qp(
        struct ibv_xrc_domain *xrc_domain,
        gasnetc_qpn_t xrc_qp_num,
        struct ibv_qp_attr *attr,
        enum ibv_qp_attr_mask attr_mask)
{
  int retval;

  retval = ibv_modify_xrc_rcv_qp(xrc_domain, xrc_qp_num, attr, attr_mask);

  if_pf (retval == EINVAL) {
    struct ibv_qp_attr      qp_attr;
    struct ibv_qp_init_attr qp_init_attr;
    int rc;

    rc = ibv_query_xrc_rcv_qp(xrc_domain, xrc_qp_num, &qp_attr, IBV_QP_STATE, &qp_init_attr);
    if (!rc && (qp_attr.qp_state >= attr->qp_state)) {
      /* No actual error, just a race against another process */
      retval = 0;
    }
  }

  return retval;
}

/* Perform a supernode-scoped broadcast from first node of the supernode.
   Note that 'src' must be a valid address on ALL callers.
 */
static void
gasnetc_supernode_bcast(void *src, size_t len, void *dst) {
  #if GASNET_PSHM
    /* Can do w/ just supernode-scoped broadcast */
    gasneti_assert(gasneti_request_pshmnet != NULL);
    gasneti_pshmnet_bootstrapBroadcast(gasneti_request_pshmnet, src, len, dst, 0);
  #else
    /* Need global Exchange when PSHM is not available */
    /* TODO: If/when is one Bcast per supernode cheaper than 1 Exchange? */
    /* TODO: Push this case down to the Bootstrap support? */
    void *all_data = gasneti_malloc(gasneti_nodes * len);
    void *my_dst = (void *)((uintptr_t )all_data + (len * gasneti_nodemap_local[0]));
    gasneti_bootstrapExchange(src, len, all_data);
    memcpy(dst, my_dst, len);
    gasneti_free(all_data);
  #endif
}

static void
gasnetc_supernode_barrier(void) {
  #if GASNET_PSHM
    gasneti_pshmnet_bootstrapBarrier();
  #else
    gasneti_bootstrapBarrier();
  #endif
}

/* XXX: Requires that at least the first call is collective */
static char*
gasnetc_xrc_tmpname(gasnetc_lid_t mylid, int index) {
  static char *tmpdir = NULL;
  static int tmpdir_len = -1;
  static pid_t pid;
  static const char pattern[] = "/GASNETxrc-%04x%01x-%06x"; /* Max 11 + 5 + 1 + 6 + 1 = 24 */
  const int filename_len = 24;
  char *filename;

  gasneti_assert(index >= 0  &&  index <= 16);

  /* Initialize tmpdir and pid only on first call */
  if (!tmpdir) {
    struct stat s;
    tmpdir = gasneti_getenv_withdefault("TMPDIR", "/tmp");
    if (stat(tmpdir, &s) || !S_ISDIR(s.st_mode)) {
      gasneti_fatalerror("XRC support requires valid $TMPDIR or /tmp");
    }
    tmpdir_len = strlen(tmpdir);

    /* Get PID of first proc per supernode */
    pid = getpid(); /* Redundant, but harmless on other processes */
    gasnetc_supernode_bcast(&pid, sizeof(pid), &pid);
  }

  filename = gasneti_malloc(tmpdir_len + filename_len);
  strcpy(filename, tmpdir);
  sprintf(filename + tmpdir_len, pattern,
          (unsigned int)(mylid & 0xffff),
          (unsigned int)(index & 0xf),
          (unsigned int)(pid & 0xffffff));
  gasneti_assert(strlen(filename) < (tmpdir_len + filename_len));

  return filename;
}

/* Create an XRC domain per HCA (once per supernode) and a shared memory file */
/* XXX: Requires that the call is collective */
extern int
gasnetc_xrc_init(void) {
  const gasnetc_lid_t mylid = gasnetc_port_tbl[0].port.lid;
  char *filename[GASNETC_IB_MAX_HCAS+1];
  size_t flen;
  int index, fd;

  /* Use per-supernode filename to create common XRC domain once per HCA */
  GASNETC_FOR_ALL_HCA_INDEX(index) {
    gasnetc_hca_t *hca = &gasnetc_hca[index];
    filename[index] = gasnetc_xrc_tmpname(mylid, index);
    fd = open(filename[index], O_CREAT, S_IWUSR|S_IRUSR);
    if (fd < 0) {
      gasneti_fatalerror("failed to create xrc domain file '%s': %d:%s", filename[index], errno, strerror(errno));
    }
    hca->xrc_domain = ibv_open_xrc_domain(hca->handle, fd, O_CREAT);
    GASNETC_VAPI_CHECK_PTR(hca->xrc_domain, "from ibv_open_xrc_domain()");
    (void) close(fd);
  }

  /* Use one more per-supernode filename to create common shared memory file */
  /* TODO: Should PSHM combine this w/ the AM segment? */
  gasneti_assert(index == gasnetc_num_hcas);
  filename[index] = gasnetc_xrc_tmpname(mylid, index);
  fd = open(filename[index], O_CREAT|O_RDWR, S_IWUSR|S_IRUSR);
  if (fd < 0) {
    gasneti_fatalerror("failed to create xrc shared memory file '%s': %d:%s", filename[index], errno, strerror(errno));
  }
  flen = GASNETI_PAGE_ALIGNUP(sizeof(gasnetc_qpn_t) * gasneti_nodes * gasnetc_alloc_qps);
  if (ftruncate(fd, flen) < 0) {
    gasneti_fatalerror("failed to resize xrc shared memory file '%s': %d:%s", filename[index], errno, strerror(errno));
  }
  /* XXX: Is there anything else that can/should be packed into the same shared memory file? */
  gasnetc_xrc_rcv_qpn = (gasnetc_qpn_t *)mmap(NULL, flen, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (gasnetc_xrc_rcv_qpn == MAP_FAILED) {
    gasneti_fatalerror("failed to mmap xrc shared memory file '%s': %d:%s", filename[index], errno, strerror(errno));
  }
  (void) close(fd);

  /* Clean up once everyone is done w/ all files */
  gasnetc_supernode_barrier();
  GASNETC_FOR_ALL_HCA_INDEX(index) {
    (void)unlink(filename[index]); gasneti_free(filename[index]);
  }
  gasneti_assert(index == gasnetc_num_hcas);
  (void)unlink(filename[index]); gasneti_free(filename[index]);

  /* Allocate SND QP table */
  gasnetc_xrc_snd_qp = gasneti_calloc(gasneti_nodemap_global_count * gasnetc_alloc_qps,
                                      sizeof(gasnetc_xrc_snd_qp_t));

  return GASNET_OK;
}
#endif /* GASNETC_IBV_XRC */

/* Distribute the qps to each peer round-robin over the ports.
   Returns NULL for cases that should not have any connection */
static const gasnetc_port_info_t *
gasnetc_select_port(gasnet_node_t node, int qpi) {
    if (gasnetc_non_ib(node)) {
      return NULL;
    }
    if (GASNETC_QPI_IS_REQ(qpi)) {
      /* Second half of table (if any) duplicates first half. */
      qpi -= gasnetc_num_qps;
    }
    
    /* XXX: At some point we lost any attempt at true "round-robin"
     * that could have balanced port use when (num_qps % num_ports) != 0.
     * The problem there was that getting a distribution that both ends
     * agreed to was a big mess.  Perhaps we bring that idea back later.
     * However, for now the same sequence of gasnetc_num_qps values is
     * repeated for each node.
     * XXX: If this changes, gasnetc_sndrcv_limits() must change to match.
     */
    return  &gasnetc_port_tbl[qpi % gasnetc_num_ports];
}

/* Setup ports array for one node */
static int
gasnetc_setup_ports(gasnetc_conn_info_t *conn_info)
{
  static const gasnetc_port_info_t **ports = NULL;

  if_pf (!ports) {
    /* Distribute the qps over the ports, same for each node. */
    int qpi;
    ports = gasneti_malloc(gasnetc_alloc_qps * sizeof(gasnetc_port_info_t *));
    for (qpi = 0; qpi < gasnetc_num_qps; ++qpi) {
      ports[qpi] = gasnetc_select_port(conn_info->node, qpi);
    #if GASNETC_IBV_SRQ
      if (gasnetc_use_srq) {
        /* Second half of table (if any) duplicates first half.
           This might NOT be the same as extending the loop */
        ports[qpi + gasnetc_num_qps] = ports[qpi];
      }
    #endif
    }
  }

  conn_info->port = ports;
  return GASNET_OK;
}

/* Create and destroy QPs to determine the inline data limit */
static void
gasnetc_check_inline_limit(int port_num, int send_wr, int send_sge)
{
  const gasnetc_port_info_t *port = &gasnetc_port_tbl[port_num];
  gasnetc_hca_t *hca = &gasnetc_hca[port->hca_index];
  gasnetc_qp_hndl_t qp_handle;

#if GASNET_CONDUIT_VAPI
  {
    VAPI_qp_init_attr_t qp_init_attr;
    VAPI_qp_prop_t      qp_prop;

    qp_init_attr.cap.max_oust_wr_rq = gasnetc_am_oust_pp * 2;
    qp_init_attr.cap.max_oust_wr_sq = send_wr;
    qp_init_attr.cap.max_sg_size_rq = 1;
    qp_init_attr.cap.max_sg_size_sq = send_sge;
    qp_init_attr.rdd_hndl           = 0;
    qp_init_attr.rq_sig_type        = VAPI_SIGNAL_REQ_WR;
    qp_init_attr.sq_sig_type        = VAPI_SIGNAL_REQ_WR;
    qp_init_attr.ts_type            = VAPI_TS_RC;
    qp_init_attr.pd_hndl            = hca->pd;
    qp_init_attr.rq_cq_hndl         = hca->rcv_cq;
    qp_init_attr.sq_cq_hndl         = hca->snd_cq;

    (void) VAPI_create_qp(hca->handle, &qp_init_attr, &qp_handle, &qp_prop);
    (void) VAPI_destroy_qp(hca->handle, qp_handle);

    gasnetc_inline_limit = MIN(gasnetc_inline_limit, qp_prop.cap.max_inline_data_sq);
  }
#else
  {
    struct ibv_qp_init_attr qp_init_attr;

    qp_init_attr.cap.max_send_wr     = send_wr;
    qp_init_attr.cap.max_recv_wr     = gasnetc_use_srq ? 0 : gasnetc_am_oust_pp * 2;
    qp_init_attr.cap.max_send_sge    = send_sge;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.qp_context          = NULL; /* XXX: Can/should we use this? */
  #if GASNETC_IBV_XRC
    qp_init_attr.qp_type             = gasnetc_use_xrc ? IBV_QPT_XRC : IBV_QPT_RC;
  #else
    qp_init_attr.qp_type             = IBV_QPT_RC;
  #endif
    qp_init_attr.sq_sig_all          = 1; /* XXX: Unless we drop 1-to-1 WQE/CQE relationship */
    qp_init_attr.srq                 = NULL; /* Should not influence inline data */
    qp_init_attr.send_cq             = hca->snd_cq;
    qp_init_attr.recv_cq             = hca->rcv_cq;

  #if GASNETC_IBV_XRC
    if (gasnetc_use_xrc) {
      qp_init_attr.xrc_domain = hca->xrc_domain;
    }
  #endif
  
    /* TODO: Binary search? */
    while (1) { /* No query for max_inline_data limit */
      qp_init_attr.cap.max_inline_data = gasnetc_inline_limit;
      qp_handle = ibv_create_qp(hca->pd, &qp_init_attr);
      if (qp_handle != NULL) break;
      if (qp_init_attr.cap.max_inline_data == -1) {
        /* Automatic max not working, fall back on manual search */
        gasnetc_inline_limit = 1024;
        continue;
      }
      if ((errno != EINVAL) || (gasnetc_inline_limit == 0)) {
        GASNETC_VAPI_CHECK_PTR(qp_handle, "from ibv_create_qp()");
        /* NOT REACHED */
      }
      gasnetc_inline_limit = MIN(1024, gasnetc_inline_limit - 1);
      /* Try again */
    }
    if (qp_init_attr.cap.max_inline_data < gasnetc_inline_limit) {
      /* Use returned value, which might be zero */
      gasnetc_inline_limit = qp_init_attr.cap.max_inline_data;
    }
    (void) ibv_destroy_qp(qp_handle);
  }
#endif
}

/* Create endpoint(s) for a given peer
 * Outputs the qpn values in the array provided.
 */
static int
gasnetc_qp_create(gasnetc_conn_info_t *conn_info)
{
    gasnetc_cep_t *cep;
    int qpi;

#if GASNET_CONDUIT_VAPI
    VAPI_qp_init_attr_t qp_init_attr;
    VAPI_qp_prop_t      qp_prop;
    int rc;

    qp_init_attr.cap.max_oust_wr_rq = gasnetc_am_oust_pp * 2;
    qp_init_attr.cap.max_oust_wr_sq = gasnetc_op_oust_pp;
    qp_init_attr.cap.max_sg_size_rq = 1;
    qp_init_attr.cap.max_sg_size_sq = GASNETC_SND_SG;
    qp_init_attr.rdd_hndl           = 0;
    qp_init_attr.rq_sig_type        = VAPI_SIGNAL_REQ_WR;
    qp_init_attr.sq_sig_type        = VAPI_SIGNAL_REQ_WR;
    qp_init_attr.ts_type            = VAPI_TS_RC;

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
      const gasnetc_port_info_t *port = conn_info->port[qpi];
      gasnetc_hca_t *hca = &gasnetc_hca[port->hca_index];

      cep->hca = hca;
      cep->hca_handle = hca->handle;
    #if GASNETC_IB_MAX_HCAS > 1
      cep->hca_index = hca->hca_index;
    #endif
      cep->sq_sema_p = &gasnetc_zero_sema;

      qp_init_attr.pd_hndl         = hca->pd;
      qp_init_attr.rq_cq_hndl      = hca->rcv_cq;
      qp_init_attr.sq_cq_hndl      = hca->snd_cq;
      rc = VAPI_create_qp(hca->handle, &qp_init_attr, &cep->qp_handle, &qp_prop);
      GASNETC_VAPI_CHECK(rc, "from VAPI_create_qp()");
      gasneti_assert(qp_prop.cap.max_oust_wr_rq >= gasnetc_am_oust_pp * 2);
      gasneti_assert(qp_prop.cap.max_oust_wr_sq >= gasnetc_op_oust_pp);
      gasneti_assert(qp_prop.cap.max_inline_data_sq >= gasnetc_inline_limit);

      conn_info->local_qpn[qpi] = qp_prop.qp_num;
    }
#else
    struct ibv_qp_init_attr     qp_init_attr;
    const int                   max_recv_wr = gasnetc_use_srq ? 0 : gasnetc_am_oust_pp * 2;
    int                         max_send_wr = gasnetc_op_oust_pp;
  #if GASNETC_IBV_XRC
    const gasnet_node_t         node = conn_info->node;
    gasnetc_xrc_snd_qp_t       *xrc_snd_qp = GASNETC_NODE2SND_QP(node);
  #endif

    qp_init_attr.cap.max_inline_data = gasnetc_inline_limit;
    qp_init_attr.cap.max_send_wr     = max_send_wr;
    qp_init_attr.cap.max_recv_wr     = max_recv_wr;
    qp_init_attr.cap.max_send_sge    = GASNETC_SND_SG;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.qp_context          = NULL; /* XXX: Can/should we use this? */
  #if GASNETC_IBV_XRC
    qp_init_attr.qp_type             = gasnetc_use_xrc ? IBV_QPT_XRC : IBV_QPT_RC;
  #else
    qp_init_attr.qp_type             = IBV_QPT_RC;
  #endif
    qp_init_attr.sq_sig_all          = 1; /* XXX: Unless we drop 1-to-1 WQE/CQE relationship */
    qp_init_attr.srq                 = NULL;

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
      gasnetc_qp_hndl_t hndl;
      const gasnetc_port_info_t *port = conn_info->port[qpi];
      gasnetc_hca_t *hca = &gasnetc_hca[port->hca_index];

      cep->hca = hca;
    #if GASNETC_IB_MAX_HCAS > 1
      cep->hca_index = hca->hca_index;
    #endif
      cep->sq_sema_p = &gasnetc_zero_sema;

      qp_init_attr.send_cq         = hca->snd_cq;
      qp_init_attr.recv_cq         = hca->rcv_cq;

    #if GASNETC_IBV_SRQ
      if (gasnetc_use_srq) {
        if (GASNETC_QPI_IS_REQ(qpi)) {
          qp_init_attr.srq = hca->rqst_srq;
          qp_init_attr.cap.max_send_wr = gasnetc_am_oust_pp;
          qp_init_attr.cap.max_send_sge = 1; /* only AMs on this QP */
        } else {
          qp_init_attr.srq = hca->repl_srq;
          qp_init_attr.cap.max_send_wr = gasnetc_op_oust_pp;
          qp_init_attr.cap.max_send_sge = GASNETC_SND_SG;
        }
        cep->srq = qp_init_attr.srq;
        max_send_wr = qp_init_attr.cap.max_send_wr;
      }
    #endif
    #if GASNETC_IBV_XRC
      if (gasnetc_use_xrc) {
        gasnetc_xrc_create_qp(hca->xrc_domain, node, qpi);

        hndl = xrc_snd_qp[qpi].handle;
        if (hndl) {
            /* per-supernode QP was already created - just reference it */
            goto finish;
        }
  
        qp_init_attr.xrc_domain = hca->xrc_domain;
        qp_init_attr.srq        = NULL;
      }
    #endif
  
      hndl = ibv_create_qp(hca->pd, &qp_init_attr);
      GASNETC_VAPI_CHECK_PTR(hndl, "from ibv_create_qp()");
      gasneti_assert(qp_init_attr.cap.max_recv_wr >= max_recv_wr);
      gasneti_assert(qp_init_attr.cap.max_send_wr >= max_send_wr);
      gasneti_assert(qp_init_attr.cap.max_inline_data >= gasnetc_inline_limit);

    #if GASNETC_IBV_XRC
      if (gasnetc_use_xrc) {
        xrc_snd_qp[qpi].handle = hndl;
        xrc_snd_qp[qpi].state = IBV_QPS_RESET;
      }

    finish:
      cep->rcv_qpn = gasnetc_use_xrc ? conn_info->local_xrc_qpn[qpi] : hndl->qp_num;
    #elif GASNETC_IBV_SRQ
      cep->rcv_qpn = hndl->qp_num;
    #endif

      cep->qp_handle = hndl;
      conn_info->local_qpn[qpi] = hndl->qp_num;
    }
#endif

    return GASNET_OK;
} /* create */

/* Advance QP state from RESET to INIT */
static int
gasnetc_qp_reset2init(gasnetc_conn_info_t *conn_info)
{
    const gasnet_node_t node = conn_info->node;
    gasnetc_qp_attr_t qp_attr;
    gasnetc_qp_mask_t qp_mask;
    gasnetc_cep_t *cep;
    int qpi;
    int rc;

#if GASNET_CONDUIT_VAPI
    VAPI_qp_cap_t       qp_cap;

    QP_ATTR_MASK_CLR_ALL(qp_mask);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_QP_STATE);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_PKEY_IX);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_PORT);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_REMOTE_ATOMIC_FLAGS);
    qp_attr.qp_state            = VAPI_INIT;
    qp_attr.pkey_ix             = 0;
    qp_attr.remote_atomic_flags = VAPI_EN_REM_WRITE | VAPI_EN_REM_READ;

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
      const gasnetc_port_info_t *port = conn_info->port[qpi];

      qp_attr.port = port->port_num;
      rc = VAPI_modify_qp(cep->hca_handle, cep->qp_handle, &qp_attr, &qp_mask, &qp_cap);
      GASNETC_VAPI_CHECK(rc, "from VAPI_modify_qp(INIT)");
    }
#else
  #if GASNETC_IBV_XRC
    gasnetc_xrc_snd_qp_t *xrc_snd_qp = GASNETC_NODE2SND_QP(node);
  #endif

    qp_mask = (enum ibv_qp_attr_mask)(IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    qp_attr.qp_state        = IBV_QPS_INIT;
    qp_attr.pkey_index      = 0;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
      const gasnetc_port_info_t *port = conn_info->port[qpi];

    #if GASNETC_IBV_SRQ
      qp_attr.qp_access_flags = GASNETC_QPI_IS_REQ(qpi)
                                    ? IBV_ACCESS_REMOTE_WRITE
                                    : IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    #endif
      qp_attr.port_num = port->port_num;

    #if GASNETC_IBV_XRC
      if (gasnetc_use_xrc) {
        rc = gasnetc_xrc_modify_qp(cep->hca->xrc_domain, conn_info->local_xrc_qpn[qpi], &qp_attr, qp_mask);
        GASNETC_VAPI_CHECK(rc, "from gasnetc_xrc_modify_qp(INIT)");
      }
    #endif

      if (GASNETC_SND_QP_NEEDS_MODIFY(xrc_snd_qp[qpi], IBV_QPS_INIT)) {
        rc = ibv_modify_qp(cep->qp_handle, &qp_attr, qp_mask);
        GASNETC_VAPI_CHECK(rc, "from ibv_modify_qp(INIT)");
      #if GASNETC_IBV_XRC
        if (gasnetc_use_xrc) xrc_snd_qp[qpi].state = IBV_QPS_INIT;
      #endif
      }
    }
#endif

    gasnetc_sndrcv_init_peer(node, conn_info->cep);

    return GASNET_OK;
} /* reset2init */

/* Advance QP state from INIT to RTR */
static int
gasnetc_qp_init2rtr(gasnetc_conn_info_t *conn_info)
{
    const gasnet_node_t node = conn_info->node;
    gasnetc_qp_attr_t qp_attr;
    gasnetc_qp_mask_t qp_mask;
    gasnetc_cep_t *cep;
    int qpi;
    int rc;

#if GASNET_CONDUIT_VAPI
    VAPI_qp_cap_t       qp_cap;

    QP_ATTR_MASK_CLR_ALL(qp_mask);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_QP_STATE);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_AV);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_PATH_MTU);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_RQ_PSN);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_QP_OUS_RD_ATOM);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_DEST_QP_NUM);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_MIN_RNR_TIMER);
    qp_attr.qp_state         = VAPI_RTR;
    qp_attr.av.sl            = 0;
    qp_attr.av.grh_flag      = 0;
    qp_attr.av.static_rate   = GASNETC_QP_STATIC_RATE;
    qp_attr.av.src_path_bits = 0;
    qp_attr.min_rnr_timer    = GASNETC_QP_MIN_RNR_TIMER;

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
      const gasnetc_port_info_t *port = conn_info->port[qpi];

      qp_attr.qp_ous_rd_atom = port->rd_atom;
      qp_attr.path_mtu       = MIN(GASNETC_QP_PATH_MTU, port->port.max_mtu);
      qp_attr.rq_psn         = GASNETC_PSN(node, qpi);
      qp_attr.av.dlid        = port->remote_lids[node];
      qp_attr.dest_qp_num    = conn_info->remote_qpn[qpi];
      rc = VAPI_modify_qp(cep->hca_handle, cep->qp_handle, &qp_attr, &qp_mask, &qp_cap);
      GASNETC_VAPI_CHECK(rc, "from VAPI_modify_qp(RTR)");
    }
#else
  #if GASNETC_IBV_XRC
    gasnetc_xrc_snd_qp_t *xrc_snd_qp = GASNETC_NODE2SND_QP(node);
  #endif

    qp_mask = (enum ibv_qp_attr_mask)(IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_DEST_QPN | IBV_QP_MIN_RNR_TIMER);
    qp_attr.qp_state         = IBV_QPS_RTR;
    qp_attr.ah_attr.sl            = 0;
    qp_attr.ah_attr.is_global     = 0;
    qp_attr.ah_attr.static_rate   = GASNETC_QP_STATIC_RATE;
    qp_attr.ah_attr.src_path_bits = 0;

    qp_attr.min_rnr_timer    = GASNETC_QP_MIN_RNR_TIMER;

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
      const gasnetc_port_info_t *port = conn_info->port[qpi];

      qp_attr.max_dest_rd_atomic = GASNETC_QPI_IS_REQ(qpi) ? 0 : port->rd_atom;
      qp_attr.path_mtu           = MIN(GASNETC_QP_PATH_MTU, port->port.max_mtu);
      qp_attr.rq_psn             = GASNETC_PSN(node, qpi);
      qp_attr.ah_attr.dlid       = port->remote_lids[node];
      qp_attr.ah_attr.port_num   = port->port_num;
      qp_attr.dest_qp_num        = conn_info->remote_qpn[qpi];

    #if GASNETC_IBV_XRC
      if (gasnetc_use_xrc) {
        rc = gasnetc_xrc_modify_qp(cep->hca->xrc_domain, cep->rcv_qpn, &qp_attr, qp_mask);
        GASNETC_VAPI_CHECK(rc, "from gasnetc_xrc_modify_qp(RTR)");

        /* The normal QP will connect, below, to the peer's XRC rcv QP */
        qp_attr.dest_qp_num = conn_info->remote_xrc_qpn[qpi];
      }
    #endif

      if (GASNETC_SND_QP_NEEDS_MODIFY(xrc_snd_qp[qpi], IBV_QPS_RTR)) {
        rc = ibv_modify_qp(cep->qp_handle, &qp_attr, qp_mask);
        GASNETC_VAPI_CHECK(rc, "from ibv_modify_qp(RTR)");
      #if GASNETC_IBV_XRC
        if (gasnetc_use_xrc) xrc_snd_qp[qpi].state = IBV_QPS_RTR;
      #endif
      }
    }
#endif

    return GASNET_OK;
} /* init2rtr */

/* Advance QP state from RTR to RTS */
static int
gasnetc_qp_rtr2rts(gasnetc_conn_info_t *conn_info)
{
    gasnetc_qp_attr_t qp_attr;
    gasnetc_qp_mask_t qp_mask;
    gasnetc_cep_t *cep;
    int qpi;
    int rc;

#if GASNET_CONDUIT_VAPI
    VAPI_qp_cap_t       qp_cap;

    QP_ATTR_MASK_CLR_ALL(qp_mask);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_QP_STATE);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_SQ_PSN);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_TIMEOUT);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_RETRY_COUNT);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_RNR_RETRY);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_OUS_DST_RD_ATOM);
    qp_attr.qp_state         = VAPI_RTS;
    qp_attr.timeout          = gasnetc_qp_timeout;
    qp_attr.retry_count      = gasnetc_qp_retry_count;
    qp_attr.rnr_retry        = GASNETC_QP_RNR_RETRY;

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
      const gasnetc_port_info_t *port = conn_info->port[qpi];

      qp_attr.sq_psn           = GASNETC_PSN(gasneti_mynode, qpi);
      qp_attr.ous_dst_rd_atom  = port->rd_atom;
      rc = VAPI_modify_qp(cep->hca_handle, cep->qp_handle, &qp_attr, &qp_mask, &qp_cap);
      GASNETC_VAPI_CHECK(rc, "from VAPI_modify_qp(RTS)");
    }
#else
  #if GASNETC_IBV_XRC
    const gasnet_node_t node = conn_info->node;
    gasnetc_xrc_snd_qp_t *xrc_snd_qp = GASNETC_NODE2SND_QP(node);
  #endif

    qp_mask = (enum ibv_qp_attr_mask)(IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
    qp_attr.qp_state         = IBV_QPS_RTS;
    qp_attr.timeout          = gasnetc_qp_timeout;
    qp_attr.retry_cnt        = gasnetc_qp_retry_count;
    qp_attr.rnr_retry        = GASNETC_QP_RNR_RETRY;

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
    #if GASNETC_IBV_XRC
      if (gasnetc_use_xrc) {
        cep->xrc_remote_srq_num = conn_info->xrc_remote_srq_num[qpi];
      }
    #endif

      if (GASNETC_SND_QP_NEEDS_MODIFY(xrc_snd_qp[qpi], IBV_QPS_RTS)) {
        const gasnetc_port_info_t *port = conn_info->port[qpi];

        qp_attr.sq_psn           = GASNETC_PSN(gasneti_mynode, qpi);
        qp_attr.max_rd_atomic    = GASNETC_QPI_IS_REQ(qpi) ? 0 : port->rd_atom;
        rc = ibv_modify_qp(cep->qp_handle, &qp_attr, qp_mask);
        GASNETC_VAPI_CHECK(rc, "from ibv_modify_qp(RTS)");
      #if GASNETC_IBV_XRC
        if (gasnetc_use_xrc) xrc_snd_qp[qpi].state = IBV_QPS_RTS;
      #endif
      }
    }
#endif

    return GASNET_OK;
} /* rtr2rts */

/* Install Send Queue semaphore() */
static int
gasnetc_set_sq_sema(gasnetc_conn_info_t *conn_info)
{
    gasnetc_cep_t *cep;
    int qpi;
  #if GASNETC_IBV_XRC
    const gasnet_node_t node = conn_info->node;
    gasnetc_xrc_snd_qp_t *xrc_snd_qp = GASNETC_NODE2SND_QP(node);
  #endif

    GASNETC_FOR_EACH_QPI(conn_info, qpi, cep) {
      gasneti_semaphore_t *sq_sema_p;
      int max_send_wr = (gasnetc_use_srq && GASNETC_QPI_IS_REQ(qpi))
                          ? gasnetc_am_oust_pp : gasnetc_op_oust_pp;

    #if GASNETC_IBV_XRC
      if (gasnetc_use_xrc) {
        sq_sema_p = xrc_snd_qp[qpi].sq_sema_p;
        if (sq_sema_p) goto skip_init;

        xrc_snd_qp[qpi].sq_sema_p = sq_sema_get();
        sq_sema_p = xrc_snd_qp[qpi].sq_sema_p;
      } else
    #endif
      {
        sq_sema_p = sq_sema_get();
      }


      /* XXX: When could/should we use the *allocated* length? */
      gasneti_semaphore_init(sq_sema_p, max_send_wr, max_send_wr);

    #if GASNETC_IBV_XRC
    skip_init:
    #endif
      gasneti_sync_writes();
      cep->sq_sema_p = sq_sema_p;
    }

    return GASNET_OK;
} /* set_qp_sema */

#if GASNETC_DYNAMIC_CONNECT

/* ------------------------------------------------------------------------------------ */
/* Support for UD endpoint used for dynamic connection setup */

/* UD recv */
typedef struct {
  gasnetc_rcv_wr_t wr;
  gasnetc_sge_t sg;
} gasnetc_ud_rcv_desc_t;

/* UD send */
typedef struct {
  gasnetc_ah_t *ah; /* shared space w/ freelist link ptr */
  gasnetc_snd_wr_t wr;
  gasnetc_sge_t sg;
} gasnetc_ud_snd_desc_t;

static gasnetc_qpn_t *conn_remote_ud_qpn = NULL;
static gasneti_lifo_head_t conn_snd_freelist = GASNETI_LIFO_INITIALIZER;

/* TODO: group the following into a UD-endpoint struct */
static gasnetc_qp_hndl_t conn_ud_qp = GASNETC_IB_CHOOSE(VAPI_INVAL_HNDL, NULL);
static gasnetc_port_info_t *conn_ud_port = NULL;
static gasnetc_hca_t *conn_ud_hca = NULL;
static int conn_ud_msg_sz = -1;

#define GASNETC_GRH_SIZE 40 /* Global Route Header is always 40 bytes */

/* TODO: If this continues to be a performance bottleneck then
 * consider cacheing since the AH is per Destination LID (the
 * target HCA's equivalent of a "MAC"), rather than per process.
 * Therefore reuse is possible w/ SMP/multi-core nodes.
 */
static gasnetc_ah_t *
gasnetc_create_ah(gasnet_node_t node)
{
  gasnetc_ah_attr_t ah_attr;
  gasnetc_ah_t *result;
	 
  result = gasneti_calloc(1, sizeof(gasnetc_ah_t));
  gasneti_weakatomic_set(&result->ref_count, 1, 0);

#if GASNET_CONDUIT_VAPI
  {
    int vstat;

    ah_attr.grh_flag      = 0;
    ah_attr.sl            = 0;
    ah_attr.src_path_bits = 0;
    ah_attr.dlid          = conn_ud_port->remote_lids[node];
    ah_attr.port          = conn_ud_port->port_num;

    vstat = VAPI_create_addr_hndl(conn_ud_hca->handle, conn_ud_hca->pd, &ah_attr, &result->ib_ah);
    GASNETC_VAPI_CHECK(vstat, "from VAPI_create_addr_hndl()");
  }
#else
  {
    ah_attr.is_global     = 0;
    ah_attr.sl            = 0;
    ah_attr.src_path_bits = 0;
    ah_attr.dlid          = conn_ud_port->remote_lids[node];
    ah_attr.port_num      = conn_ud_port->port_num;

    result->ib_ah = ibv_create_ah(conn_ud_hca->pd, &ah_attr);
    GASNETC_VAPI_CHECK_PTR(result, "from ibv_create_ah()");
  }
#endif

  return result;
}

static void
gasnetc_put_ah(gasnetc_ah_t *ah)
{
  if (gasneti_weakatomic_decrement_and_test(&ah->ref_count, 0)) {
#if GASNET_CONDUIT_VAPI
    int vstat = VAPI_destroy_addr_hndl(conn_ud_hca->handle, ah->ib_ah);
    GASNETC_VAPI_CHECK(vstat, "from VAPI_destroy_addr_hndl()");
#else
    int vstat = ibv_destroy_ah(ah->ib_ah);
    GASNETC_VAPI_CHECK(vstat, "from ibv__destroy_ah()");
#endif
    gasneti_free(ah);
  }
}

/* Post a work request to the receive queue of the UD QP */
static void
gasnetc_rcv_post_ud(gasnetc_ud_rcv_desc_t *desc)
{
  int vstat;

#if GASNET_CONDUIT_VAPI
  {
    vstat = VAPI_post_rr(conn_ud_hca->handle, conn_ud_qp, &desc->wr);
  }
#else
  {
    gasnetc_rcv_wr_t *bad_wr;
    vstat = ibv_post_recv(conn_ud_qp, &desc->wr, &bad_wr);
  }
#endif

  GASNETC_VAPI_CHECK(vstat, "while posting a UD receive work request");
}

/* Post a work request to the send queue of the UD QP */
static void
gasnetc_snd_post_ud(gasnetc_ud_snd_desc_t *desc, gasnetc_ah_t *ah, gasnet_node_t node)
{
  gasneti_semaphore_t *snd_cq_sema_p = conn_ud_hca->snd_cq_sema_p;
  gasnetc_snd_wr_t *wr = &desc->wr;
  int vstat;

  if (NULL == ah) {
    ah = gasnetc_create_ah(node);
  } else {
    gasneti_weakatomic_increment(&ah->ref_count, 0);
  }
  desc->ah = ah;

  /* Loop until space is available for 1 new entry on the CQ. */
  if_pf (!gasneti_semaphore_trydown(snd_cq_sema_p)) {
    GASNETC_TRACE_WAIT_BEGIN();
    do {
      GASNETI_WAITHOOK();
      gasnetc_sndrcv_poll(1);
    } while (!gasneti_semaphore_trydown(snd_cq_sema_p));
    GASNETC_TRACE_WAIT_END(CONN_STALL_CQ);
  }

#if GASNET_CONDUIT_VAPI
  { 
    wr->remote_qp   = conn_remote_ud_qpn[node];
    wr->remote_ah   = ah->ib_ah;
    vstat = VAPI_post_sr(conn_ud_hca->handle, conn_ud_qp, wr);
  }
#else
  {
    gasnetc_snd_wr_t *bad_wr;
    wr->wr.ud.remote_qpn  = conn_remote_ud_qpn[node];
    wr->wr.ud.ah          = ah->ib_ah;

    vstat = ibv_post_send(conn_ud_qp, wr, &bad_wr);
  }
#endif

  GASNETC_VAPI_CHECK(vstat, "while posting a UD send work request");
}

/* ------------------------------------------------------------------------------------ */

/* Sketch of the connection progress engine state space.
 *
 *                           [NONE]
 *                              |
 *                              v
 *                    +---------+---------+
 *                    |                   |
 *                    v                   |
 *                (send REQ)              |
 *                [REQ_SENT]----+         |
 *                    |         |         |
 *                    |         v         |
 *                    |    (*recv REQ)    v
 *                    v         |     (recv REP)
 *                (recv REP)    |     (send REP)
 *                [REP_RCVD]<---+---->[REP_SENT]
 *                    |                   |
 *                    v                   |
 *                (send RTU)              |
 *                [RTU_SENT]              v
 *                    |               (recv RTU)
 *                    v               (send ACK)
 *                (recv ACK)              |
 *                [ACK_RCVD]              |
 *                    |                   |
 *                    v                   v
 *                    +---------+---------+
 * Key:                         |
 * (foo) = Comm. events         v
 * [FOO] = States            [DONE]
 *
 *
 * *: When both peers begin by sending a REQ, we have the "ACTIVE-ACTIVE" case.
 * This is resolved by BOTH peers treating one of the REQs as if it were a REP
 * instead (they carry the same payload).  So, when a REQ is received in the
 * REQ_SENT state, one peer will move to REP_SENT and the other to REP_RCVD
 * (selected deterministically and without additional communication).
 *
 * If a message is lost then a resend will occur without a state change.
 * The following are the resends and are NOT depicted in the diagram above:
 *    + Timer expires in REQ_SENT state - resend the REQ msg
 *    + Timer expires in RTU_SENT state - resend the RTU msg
 *    + REQ recvd in the REP_SENT state - resend the REP msg
 *    + RTU recvd in the DONE state     - resend the ACK msg
 *    + REQ recvd in the RTU_SENT state - resend the REQ msg
 * The last one (recv REQ in RTU_SENT resends REQ) could use some explanation:
 *  In this case because we are receiving REQ (rather than REP) from our peer,
 *  we know that we are in the ACTIVE-ACTIVE case and we can reason that the
 *  peer may not realize this yet (the other option is that this REQ is just
 *  late).  If our peer never received our REQ then it cannot move forward
 *  without the data contained in a REQ or REP.  So, the resolution is to
 *  resend our REQ.
 *
 * GASNet-level state transitions:
 * + On the ACTIVE side the GASNet-level CEP is installed in the NODE2CEP
 *   table on the transition to GASNETC_CONN_STATE_RTU_SENT, but it in a
 *   receive-only state (Send Queue semaphore is zero).
 * + The ACTIVE peer transitions to the full send-and-receive state when
 *   it receives the ACK that ensures the PASSIVE peer is ready to receive.
 * + The PASSIVE peer installs the CEP in the NODE2CEP table with full
 *   capability to both send and receive when it receives the RTU, which
 *   indicates the ACTIVE peer is ready to receive.
 *
 * Note that it is possible that due to multi-threading or packet loss an
 * AM sent from the PASSIVE peer to the ACTIVE one could arrive after the
 * ACTIVE peer has sent RTU, but before it has received the ACK.  In this
 * case (and ONLY this case) it is possible for the ACTIVE peer to find
 * itself needing to send (an AMReply) even though the Send Queue semaphore
 * is zero!  Fortunately, this case is easily detected in the code that
 * deals with the failed semaphore_try_down() and is already outside the
 * normal "fast path".  The function gasnetc_conn_implied_ack() is called
 * to perform the work that would otherwise have been done when the
 * poll loop in the ACTIVE peer notices the ACK.
 */

typedef enum {
  /* State valid for both the ACTIVE and PASSIVE peer: */
  GASNETC_CONN_STATE_NONE = 0, /* Newly created ACTIVE or PASSIVE */

  /* States valid only for ACTIVE peer: */
  GASNETC_CONN_STATE_REQ_SENT, /* Have sent REQ and am waiting for REP */
  GASNETC_CONN_STATE_REP_RCVD, /* Have received REP but have not yet acted on it */
  GASNETC_CONN_STATE_RTU_SENT, /* Have sent RTU and am waiting for ACK */
  GASNETC_CONN_STATE_ACK_RCVD, /* Have received ACK but have not yet acted on it */

  /* States valid only for PASIVE peer: */
  GASNETC_CONN_STATE_REP_SENT, /* Have sent REP and am waiting for RTU */

  /* State valid for both the ACTIVE and PASSIVE peer: */
  GASNETC_CONN_STATE_DONE      /* Have reached RTS - connection is ready */
} gasnetc_conn_state_t;

typedef struct gasnetc_conn_s {
  struct gasnetc_conn_s *next, *prev;
  volatile
  gasnetc_conn_state_t  state;
  gasnetc_conn_info_t   info;
  gasnetc_ah_t          *ah;
  gasneti_tick_t        xmit_time;
  gasneti_tick_t        reply_time;
#if GASNETI_STATS_OR_TRACE
  gasneti_tick_t         start_time;
  int                    start_active;
#endif
  int                    ref_count;
} gasnetc_conn_t;

typedef enum {
  GASNETC_CONN_CMD_REQ = 1,
  GASNETC_CONN_CMD_REP,
  GASNETC_CONN_CMD_RTU,
  GASNETC_CONN_CMD_ACK,
} gasnetc_conn_cmd_t;

static gasnetc_ud_snd_desc_t *
conn_get_snd_desc(gasnetc_conn_cmd_t cmd)
{
  gasnetc_ud_snd_desc_t *desc =  gasneti_lifo_pop(&conn_snd_freelist);
  GASNETC_TRACE_WAIT_BEGIN();

  if (NULL == desc) {
    do {
      GASNETI_WAITHOOK();
      gasnetc_sndrcv_poll(1);
      desc = gasneti_lifo_pop(&conn_snd_freelist);
    } while (NULL == desc);
    GASNETC_TRACE_WAIT_END(CONN_STALL_DESC);
  }
  desc->wr.imm_data = cmd | (gasneti_mynode << 16);
  return desc;
}

static void
conn_send_data(gasnetc_conn_t *conn, gasnetc_conn_cmd_t cmd)
{
  gasnetc_conn_info_t *conn_info = &conn->info;
  gasnetc_ud_snd_desc_t *desc = conn_get_snd_desc(cmd);
  void *buf = (void *)(uintptr_t)desc->sg.addr;

#if GASNETC_IBV_XRC
  if (gasnetc_use_xrc) {
    gasnetc_xrc_conn_data_t *data = (gasnetc_xrc_conn_data_t *)buf;
    int qpi;

    for (qpi = 0; qpi < gasnetc_alloc_qps; ++qpi) {
      gasnetc_hca_t *hca = conn_info->cep[qpi].hca;
      struct ibv_srq *srq = GASNETC_QPI_IS_REQ(qpi) ? hca->rqst_srq : hca->repl_srq;
      data[qpi].srq_num = srq->xrc_srq_num;
      data[qpi].xrc_qpn = conn_info->local_xrc_qpn[qpi];
      data[qpi].qpn     = conn_info->local_qpn[qpi];
    }
  } else
#endif
  {
    memcpy(buf, conn_info->local_qpn, conn_ud_msg_sz);
  }
    
  desc->sg.gasnetc_f_sg_len = conn_ud_msg_sz;
  gasnetc_snd_post_ud(desc, conn->ah, conn_info->node);
}

static void
conn_send_empty(gasnetc_ah_t *ah, gasnet_node_t node, gasnetc_conn_cmd_t cmd)
{
  gasnetc_ud_snd_desc_t *desc = conn_get_snd_desc(cmd);

  desc->sg.gasnetc_f_sg_len = GASNETC_ALLOW_0BYTE_MSG ? 0 : 1;
  gasnetc_snd_post_ud(desc, ah, node);
}

/* ------------------------------------------------------------------------------------ */

/* Create UD QP and advance all the way to RTS */
static int
gasnetc_qp_setup_ud(gasnetc_port_info_t *port, int fully_connected)
{
  #if GASNET_CONDUIT_VAPI
    VAPI_qp_cap_t qp_cap;
  #endif
    gasnetc_qpn_t gasnetc_conn_qpn = 0;
    gasnetc_qp_attr_t qp_attr;
    gasnetc_qp_mask_t qp_mask;
    gasnetc_memreg_t mem_reg;
    uint32_t my_qkey = 0;
    uintptr_t addr;
    int rc;

    const int max_recv_wr = gasnetc_ud_rcvs;
    const int max_send_wr = gasnetc_ud_snds;

  #if GASNETC_IBV_XRC
    const int send_sz = gasnetc_alloc_qps *
                            (gasnetc_use_xrc ? sizeof(gasnetc_xrc_conn_data_t)
                                             : sizeof(gasnetc_qpn_t));
  #else
    const int send_sz = gasnetc_alloc_qps * sizeof(gasnetc_qpn_t);
  #endif
  #if 0
    /* XXX: This is the correct value. */
    const int recv_sz = send_sz + GASNETC_GRH_SIZE; /* recv size sees 40-byte GRH */
  #else
    /* XXX: For unknown reason InfiniPath may need 4 EXTRA bytes or rcvr will drop. */
    const int recv_sz = 4 + send_sz + GASNETC_GRH_SIZE; /* recv size sees 40-byte GRH */
  #endif

    /* If this node is fully connected, just participate in the qpn Exchange,
     * but don't allocate any resources for the UD communications.
     */
    if (fully_connected) {
      gasnetc_qpn_t *tmp = gasneti_malloc(gasneti_nodes * sizeof(gasnetc_qpn_t));
      gasneti_assert(gasnetc_conn_qpn == 0);
      gasneti_bootstrapExchange(&gasnetc_conn_qpn, sizeof(gasnetc_conn_qpn), tmp);
      gasneti_free(tmp);
      return GASNET_OK;
    }

    /* Record frequently needed paramaters */
    conn_ud_port = port;
    conn_ud_hca = &gasnetc_hca[port->hca_index];
    conn_ud_msg_sz = send_sz;

    /* CREATE */
#if GASNET_CONDUIT_VAPI
  {
    VAPI_qp_init_attr_t qp_init_attr;
    VAPI_qp_prop_t      qp_prop;

    qp_init_attr.cap.max_oust_wr_sq = max_send_wr;
    qp_init_attr.cap.max_oust_wr_rq = max_recv_wr;
    qp_init_attr.cap.max_sg_size_sq = 1;
    qp_init_attr.cap.max_sg_size_rq = 1;
    qp_init_attr.rdd_hndl           = 0;
    qp_init_attr.rq_sig_type        = VAPI_SIGNAL_REQ_WR;
    qp_init_attr.sq_sig_type        = VAPI_SIGNAL_REQ_WR;
    qp_init_attr.ts_type            = VAPI_TS_UD;
    qp_init_attr.pd_hndl            = conn_ud_hca->pd;
    qp_init_attr.rq_cq_hndl         = conn_ud_hca->rcv_cq;
    qp_init_attr.sq_cq_hndl         = conn_ud_hca->snd_cq;

    rc = VAPI_create_qp(conn_ud_hca->handle, &qp_init_attr, &conn_ud_qp, &qp_prop);
    if (rc != 0) return GASNET_ERR_RESOURCE;
    gasnetc_conn_qpn = qp_prop.qp_num;
  }
#else
  {
    struct ibv_qp_init_attr     qp_init_attr;

    qp_init_attr.cap.max_send_wr     = max_send_wr;
    qp_init_attr.cap.max_recv_wr     = max_recv_wr;
    qp_init_attr.cap.max_send_sge    = 1;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.qp_context          = NULL;
    qp_init_attr.qp_type             = IBV_QPT_UD;
    qp_init_attr.sq_sig_all          = 1;
    qp_init_attr.srq                 = NULL;
    qp_init_attr.send_cq             = conn_ud_hca->snd_cq;
    qp_init_attr.recv_cq             = conn_ud_hca->rcv_cq;
    qp_init_attr.cap.max_inline_data = 0;

    conn_ud_qp = ibv_create_qp(conn_ud_hca->pd, &qp_init_attr);
    if_pf (NULL == conn_ud_qp) return GASNET_ERR_RESOURCE;
    gasnetc_conn_qpn = conn_ud_qp->qp_num;
  }
#endif

    /* Exchange the qpns */
    conn_remote_ud_qpn = gasneti_malloc(gasneti_nodes * sizeof(gasnetc_qpn_t));
    gasneti_bootstrapExchange(&gasnetc_conn_qpn, sizeof(gasnetc_conn_qpn), conn_remote_ud_qpn);

    /* Generate a per-job QKey from the qpns.
     * This value is "unpredictable" but might not be unique.  In fact,
     * consecutive runs on the same nodes can produce the same value.
     * TODO: Is there some info already available to make this more unique?
     */
    { gasnet_node_t node;
      gasneti_assert(my_qkey == 0);
      for (node = 0; node < gasneti_nodes; ++node) {
        my_qkey ^= conn_remote_ud_qpn[node];
      }
    }

    /* Allocate pinned memory */
    { const size_t size = GASNETI_PAGE_ALIGNUP((max_send_wr * send_sz) + /* XXX: Omit send if inline */
                                               (max_recv_wr * recv_sz));
      void *buf = gasneti_mmap(size);
      if_pf (MAP_FAILED == buf) {
        gasneti_fatalerror("Failed to allocate memory for dynamic connection setup");
      }

      rc = gasnetc_pin(conn_ud_hca, buf, size,
                       GASNETC_ACL_LOC_WR, &mem_reg);
      GASNETC_VAPI_CHECK(rc, "while pinning memory for dynamic connection setup");

      addr = (uintptr_t)buf;
    }

    /* RESET -> INIT */
#if GASNET_CONDUIT_VAPI
    QP_ATTR_MASK_CLR_ALL(qp_mask);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_QP_STATE);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_PKEY_IX);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_PORT);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_QKEY);
    qp_attr.qp_state            = VAPI_INIT;
    qp_attr.pkey_ix             = 0;
    qp_attr.qkey                = my_qkey;
    qp_attr.port                = port->port_num;

    rc = VAPI_modify_qp(conn_ud_hca->handle, conn_ud_qp, &qp_attr, &qp_mask, &qp_cap);
    GASNETC_VAPI_CHECK(rc, "from VAPI_modify_qp(UD INIT)");
#else
    qp_mask = (enum ibv_qp_attr_mask)(IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    qp_attr.qp_state        = IBV_QPS_INIT;
    qp_attr.pkey_index      = 0;
    qp_attr.qkey            = my_qkey;
    qp_attr.port_num        = port->port_num;
    rc = ibv_modify_qp(conn_ud_qp, &qp_attr, qp_mask);
    GASNETC_VAPI_CHECK(rc, "from ibv_modify_qp(UD INIT)");
#endif

    /* Post RCVs */
    { gasnetc_ud_rcv_desc_t *desc;
      int i;

      desc = gasneti_calloc(max_recv_wr, sizeof(gasnetc_ud_rcv_desc_t));
      for (i = 0; i < max_recv_wr; ++i, ++desc, addr += recv_sz) {
        desc->wr.gasnetc_f_wr_num_sge = 1;
        desc->wr.gasnetc_f_wr_sg_list = &desc->sg;
        desc->wr.gasnetc_f_wr_id      = 1 | (uintptr_t)desc;   /* CQE will point back to this request */
      #if GASNET_CONDUIT_VAPI
        desc->wr.opcode               = VAPI_RECEIVE;
        desc->wr.comp_type            = VAPI_SIGNALED;
      #else   
        desc->wr.next                 = NULL;
      #endif  
        desc->sg.gasnetc_f_sg_len = recv_sz;
        desc->sg.addr             = addr;
        desc->sg.lkey             = mem_reg.lkey;
        gasnetc_rcv_post_ud(desc);
      }
    }

    /* INIT -> RTR */
#if GASNET_CONDUIT_VAPI
    QP_ATTR_MASK_CLR_ALL(qp_mask);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_QP_STATE);
    qp_attr.qp_state         = VAPI_RTR;
    rc = VAPI_modify_qp(conn_ud_hca->handle, conn_ud_qp, &qp_attr, &qp_mask, &qp_cap);
    GASNETC_VAPI_CHECK(rc, "from VAPI_modify_qp(UD RTR)");
#else
    qp_mask = IBV_QP_STATE;
    qp_attr.qp_state        = IBV_QPS_RTR;
    rc = ibv_modify_qp(conn_ud_qp, &qp_attr, qp_mask);
    GASNETC_VAPI_CHECK(rc, "from ibv_modify_qp(UD RTR)");
#endif

    /* RTR -> RTS */
#if GASNET_CONDUIT_VAPI
    QP_ATTR_MASK_CLR_ALL(qp_mask);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_QP_STATE);
    QP_ATTR_MASK_SET(qp_mask, QP_ATTR_SQ_PSN);
    qp_attr.qp_state         = VAPI_RTS;
    qp_attr.sq_psn           = 0xcafef00d;
    rc = VAPI_modify_qp(conn_ud_hca->handle, conn_ud_qp, &qp_attr, &qp_mask, &qp_cap);
    GASNETC_VAPI_CHECK(rc, "from VAPI_modify_qp(UD RTS)");
#else
    qp_mask = (enum ibv_qp_attr_mask)(IBV_QP_STATE | IBV_QP_SQ_PSN);
    qp_attr.qp_state        = IBV_QPS_RTS;
    qp_attr.sq_psn          = 0xcafef00d;
    rc = ibv_modify_qp(conn_ud_qp, &qp_attr, qp_mask);
    GASNETC_VAPI_CHECK(rc, "from ibv_modify_qp(UD RTS)");
#endif

    /* Create SNDs */
    { gasnetc_ud_snd_desc_t *desc;
      int i;

      desc = gasneti_calloc(max_send_wr, sizeof(gasnetc_ud_snd_desc_t));
      for (i = 0; i < max_send_wr; ++i, ++desc, addr += send_sz) {
        desc->wr.gasnetc_f_wr_num_sge = 1;
        desc->wr.gasnetc_f_wr_sg_list = &desc->sg;
        desc->wr.gasnetc_f_wr_id      = 1 | (uintptr_t)desc;   /* CQE will point back to this request */
        desc->wr.opcode               = GASNETC_WR_SEND_WITH_IMM;
      #if GASNET_CONDUIT_VAPI
        desc->wr.comp_type            = VAPI_SIGNALED;
        desc->wr.set_se               = 0;
        desc->wr.fence                = 0;
        desc->wr.remote_qkey          = my_qkey;
      #else   
        desc->wr.send_flags           = (enum ibv_send_flags)0;
        desc->wr.next                 = NULL;
        desc->wr.wr.ud.remote_qkey    = my_qkey;
      #endif  
      #if GASNET_DEBUG
        desc->sg.gasnetc_f_sg_len = ~0;
      #endif  
        desc->sg.addr             = addr;
        desc->sg.lkey             = mem_reg.lkey;
        gasneti_lifo_push(&conn_snd_freelist, desc);
      }
    }

    /* "warmup" the timers to ensure we don't pay the potentially high cost
       of the tick normalization while trying to manage retransmissions. */
    (void) gasneti_tick_granularity();

    return GASNET_OK;
} /* setup_ud */

/* ------------------------------------------------------------------------------------ */

static gasneti_mutex_t gasnetc_conn_tbl_lock = GASNETI_MUTEX_INITIALIZER;
static gasnetc_conn_t *gasnetc_conn_tbl = NULL;

static gasnetc_conn_t *
gasnetc_get_conn(gasnet_node_t node)
{
  gasnetc_conn_t *conn = gasnetc_conn_tbl;

  while (conn && (conn->info.node != node)) {
    conn = conn->next;
  }

  if (conn) {
    /* Found it - nothing more to do */
  } else if (GASNETC_NODE2CEP(node)) {
    /* Connection complet(ing) - nothing more to do */
  } else {
    /* Create new */
    conn = gasneti_malloc(sizeof(*conn));
    conn->next = gasnetc_conn_tbl;
    conn->prev = NULL;
    if (gasnetc_conn_tbl) {
      gasnetc_conn_tbl->prev = conn;
    }
    gasnetc_conn_tbl = conn;

  #if GASNETI_STATS_OR_TRACE
    conn->start_time = GASNETI_TICKS_NOW_IFENABLED(C);
    conn->start_active = 0;
  #endif

    conn->state = GASNETC_CONN_STATE_NONE;
    conn->info.node = node;
    conn->info.cep = (gasnetc_cep_t *)
                       gasnett_malloc_aligned(GASNETI_CACHE_LINE_BYTES,
                                              gasnetc_alloc_qps * sizeof(gasnetc_cep_t));
    memset(conn->info.cep, 0, gasnetc_alloc_qps * sizeof(gasnetc_cep_t));
    conn->info.local_qpn = gasneti_malloc(2 * gasnetc_alloc_qps * sizeof(gasnetc_qpn_t));
    conn->info.remote_qpn = conn->info.local_qpn + gasnetc_alloc_qps;
  #if GASNETC_IBV_XRC
    if (gasnetc_use_xrc) {
      conn->info.local_xrc_qpn = &gasnetc_xrc_rcv_qpn[node * gasnetc_alloc_qps];
      conn->info.remote_xrc_qpn = gasneti_malloc(gasnetc_alloc_qps * sizeof(gasnetc_qpn_t));
      conn->info.xrc_remote_srq_num = gasneti_malloc(gasnetc_alloc_qps * sizeof(uint32_t));
    }
  #endif
    gasnetc_setup_ports(&conn->info);
    conn->ah = gasnetc_create_ah(node);
    conn->ref_count = 1;
  }

  return conn;
}

static void
gasnetc_put_conn(gasnetc_conn_t *conn)
{
  if (--conn->ref_count) return;

  if (conn->next) {
    conn->next->prev = conn->prev;
  }
  if (conn->prev) {
    conn->prev->next = conn->next;
  } else {
    gasnetc_conn_tbl = conn->next;
  }
  gasneti_free(conn->info.local_qpn);
#if GASNETC_IBV_XRC
  if (gasnetc_use_xrc) {
    gasneti_free(conn->info.remote_xrc_qpn);
    gasneti_free(conn->info.xrc_remote_srq_num);
  }
#endif
  gasnetc_put_ah(conn->ah);
  gasneti_free(conn);
}

/* Defaults:
 *   Min is 1ms which at least 25% more than the ENTIRE connect should take.
 *   Max is (1 << 24) us, which, doubling from an inital min=1ms, means 16.384s.
 * Sum is upto 32s spent retrying EACH of the two round-trip message exchanges.
 *
 * TODO: It *might* be helpful to "learn" retransmit intervals dynamically.
 * Here are some notes based on initial experiments run on Carver.
 * + The number of "packets" sent to an given peer is just 2, making tuning per
 *   peer nearly pointless.
 * + If pooling the RTT data across all connections then the number of packets may
 *   often be too small to gain statistically meaningful RTT information, given the
 *   high variance that will result from pooling info from multiple nodes.
 * + If pooling info per-supernode the data may still be high-variance because the
 *   remote attentiveness is a factor in the RTT.
 * + Summary of previous 3 items: not clear that smoothed-RTT estimators are useful.
 * + IF using SRTT estimators, then we should follow advice given in RFC 1122:
 *   - Use Karn's algorithm (but see below) to 1) use only unambiguous RTT values in
 *     updating the SRTT, and 2) carry-over backed-off RTO to next packet when no
 *     updated SRTT is available to compute a "fresh" RTO.
 *   - Use Van Jacobson's variance-aware RTO computation.  However, we should probably
 *     use the original (a+2v) version, not the (a+4v) version that was updated based
 *     on behavior of TCP slow-start over SLIP.  The reasoning is 2-fold: we are NOT
 *     dealing with bandwidth-dominated links; and our variance is already going to
 *     be artificially high if pooled over multiple peers.
 * + Karn's algorithm says to discard any RTT measurement for a packet that has been
 *   retransmitted, because we cannot know if the response is to the original or one
 *   of the resends.  However, unlike TCP we have the option to put info in the header
 *   that would allow us to keep Karn's rule about using only "unambiguous" RTT
 *   measurements while admitting a larger set of measurements.  Some options:
 *   - A single header bit would distinguish the original packet and replies to it.  This
 *     allows us to use RTT for any replies to the original, even if there were some
 *     resends.  This could help to more quickly raise a SRTT estimate that is too low.
 *   - A counter in the header could give the sequence for each resend, and be echoed
 *     back in the reply.  This could easily be used to distinguish not only replies
 *     to the first send (as with the 1-bit modification), but also replies to the
 *     most recent resend.  Since both the first and last send timestamps are on hand
 *     at the sender, unambiguous RTT computations would be possible for both these
 *     cases.  Would only need 4 or 5 bits, and at least 8 are easily available.
 *   - Using the same counter as the previous item with the addition of a "log" of the
 *     (re)send times would allow unambiguous RTT computation regardless of which of
 *     multiple sends was the one to receive a reply.
 *   - Sending a timestamp on a request (again echoed back in the reply) would allow
 *     unambiguous RTT computation w/o the log, but at the expense of up to an 8-byte
 *     timestamp.  For the non-SRQ case current payload is only 4 bytes, making this
 *     timestamp a non-trivial expansion of the payload.  The log feels "cheaper".
 * + Use of the resend-sequence number mentioned above would allow use to independently
 *   estimate loss rate in addition to RTT time, with the caveat that we cannot be
 *   sure if loss was due to congestion in the fabric or overrun at the receiver.
 *   While I have not done (or searched for in literature) the corresponding analysis,
 *   I suspect that something intelligent might be done with this additional data.
 *
 * All of this may be more important if/when we want to to AM-over-UD or multicast.
 */

/* While env vars are in us, these store ns */
static uint64_t gasnetc_conn_retransmit_min_ns = 1000 * 1000;
static uint64_t gasnetc_conn_retransmit_max_ns = ((uint64_t)1 << 24) * 1000;

/* NOTE: releases and reacquires the lock */
static void
gasnetc_timed_conn_wait(gasnetc_conn_t *conn, gasnetc_conn_state_t state, 
                        void (*fn)(gasnetc_conn_t *))
{
  uint64_t timeout_ns = gasnetc_conn_retransmit_min_ns;
  gasneti_tick_t prev_time = conn->xmit_time;
#if GASNETI_STATS_OR_TRACE
  gasneti_tick_t end_time;
  int resends = 0;
#endif

  gasneti_mutex_unlock(&gasnetc_conn_tbl_lock);
  while (1) {
    while ((conn->state == state) &&
           (gasneti_ticks_to_ns(gasneti_ticks_now() - prev_time) < timeout_ns)) {
      GASNETI_WAITHOOK();
      gasnetc_sndrcv_poll(0); /* works even before _attach */
    }
  #if GASNETI_STATS_OR_TRACE
    end_time = GASNETI_TICKS_NOW_IFENABLED(C);
  #endif

    if (conn->state != state) break; /* Done */

    timeout_ns *= 2;
    if (timeout_ns > gasnetc_conn_retransmit_max_ns) break; /* limit reached */

    /* retransmit */
    (*fn)(conn);
    prev_time = gasneti_ticks_now();
  #if GASNETI_STATS_OR_TRACE
    ++resends;
  #endif
  }
  gasneti_mutex_lock(&gasnetc_conn_tbl_lock);

  if (conn->state == state) {
    gasneti_fatalerror("Node %d timed out attempting dynamic connection to node %d (state = %d)",
                       (int)gasneti_mynode, (int)conn->info.node, state);
  }

#if GASNETI_STATS_OR_TRACE
  switch(state) {
  case GASNETC_CONN_STATE_REQ_SENT:
    GASNETI_TRACE_EVENT_TIME(C, CONN_REQ2REP, (end_time - conn->xmit_time));
    GASNETC_STAT_EVENT_VAL(CONN_REQ, resends);
    break;
  case GASNETC_CONN_STATE_RTU_SENT:
    GASNETI_TRACE_EVENT_TIME(C, CONN_RTU2ACK, (end_time - conn->xmit_time));
    GASNETC_STAT_EVENT_VAL(CONN_RTU, resends);
    break;
  default:
    break;
  }
#endif
}

static void
conn_send_req(gasnetc_conn_t *conn)
{
  conn_send_data(conn, GASNETC_CONN_CMD_REQ);
}

static void
conn_send_rtu(gasnetc_conn_t *conn)
{
  conn_send_empty(conn->ah, conn->info.node, GASNETC_CONN_CMD_RTU);
}

static void
conn_send_rep(gasnetc_conn_t *conn)
{
  conn_send_data(conn, GASNETC_CONN_CMD_REP);
  GASNETC_STAT_EVENT(CONN_REP);
}

static void
conn_send_ack(gasnetc_conn_t *conn, gasnet_node_t node)
{
  conn_send_empty(conn ? conn->ah : NULL, node, GASNETC_CONN_CMD_ACK);
  GASNETC_STAT_EVENT(CONN_ACK);
}

/* "wrapper" to centralize tracing/stats */
#if GASNETI_STATS_OR_TRACE
GASNETI_INLINE(gasnetc_dynamic_done)
void gasnetc_dynamic_done(gasnetc_conn_t *conn, int active)
{
  GASNETC_STAT_EVENT(CONN_DYNAMIC);
  if (active) {
    GASNETI_TRACE_EVENT_TIME(C, CONN_TIME_ACTV, (gasneti_ticks_now() - conn->start_time));
    GASNETI_TRACE_PRINTF(C, ("Dynamic connection to node %d", (int)conn->info.node));
  } else if (conn->start_active) {
    GASNETI_TRACE_EVENT_TIME(C, CONN_TIME_A2P, (gasneti_ticks_now() - conn->start_time));
    GASNETI_TRACE_PRINTF(C, ("Dynamic connection with node %d", (int)conn->info.node));
  } else {
    GASNETI_TRACE_EVENT_TIME(C, CONN_TIME_PASV, (gasneti_ticks_now() - conn->start_time));
    GASNETI_TRACE_PRINTF(C, ("Dynamic connection from node %d", (int)conn->info.node));
  }
}
#else
#define gasnetc_dynamic_done(c,a) ((void)0)
#endif

extern gasnetc_cep_t *
gasnetc_connect_to(gasnet_node_t node)
{
  gasnetc_cep_t *result = NULL;

  gasneti_mutex_lock(&gasnetc_conn_tbl_lock);
  do {
    gasnetc_conn_t *conn = gasnetc_get_conn(node);

    if (!conn || (conn->state != GASNETC_CONN_STATE_NONE)) {
      /* We are not the first to request this connection */
      break;
    }
  #if GASNETI_STATS_OR_TRACE
    conn->start_active = 1;
  #endif

    (void) gasnetc_qp_create(&conn->info);
    conn->state = GASNETC_CONN_STATE_REQ_SENT;

    conn_send_req(conn);
    conn->xmit_time = gasneti_ticks_now();

    (void) gasnetc_qp_reset2init(&conn->info);
    gasnetc_timed_conn_wait(conn, GASNETC_CONN_STATE_REQ_SENT, &conn_send_req);

    if ((conn->state == GASNETC_CONN_STATE_REP_SENT) ||
        (conn->state == GASNETC_CONN_STATE_DONE)) {
      /* Resolved the active-active case by becoming the Passive peer */
      gasnetc_put_conn(conn);
      break;
    }
    gasneti_assert(conn->state == GASNETC_CONN_STATE_REP_RCVD);

    (void) gasnetc_qp_init2rtr(&conn->info);
    gasneti_sync_writes(); /* "finalize" cep data */
    GASNETC_NODE2CEP(node) = conn->info.cep;
    conn->state = GASNETC_CONN_STATE_RTU_SENT;

    conn_send_rtu(conn);
    conn->xmit_time = gasneti_ticks_now();

    gasnetc_sndrcv_attach_peer(node, conn->info.cep);
    (void) gasnetc_qp_rtr2rts(&conn->info);
    gasnetc_timed_conn_wait(conn, GASNETC_CONN_STATE_RTU_SENT, &conn_send_rtu);

    if (conn->state != GASNETC_CONN_STATE_DONE) {
      gasneti_assert(conn->state == GASNETC_CONN_STATE_ACK_RCVD);
      (void) gasnetc_set_sq_sema(&conn->info);
      conn->state = GASNETC_CONN_STATE_DONE;
      gasnetc_dynamic_done(conn, 1);
    } else {
      /* Connection completed by gasnetc_conn_implied_ack() */
    }

    gasnetc_put_conn(conn);
  } while (0);
  gasneti_mutex_unlock(&gasnetc_conn_tbl_lock);

  gasneti_polluntil(NULL != (result = GASNETC_NODE2CEP(node)));

  return result;
}

extern void
gasnetc_conn_implied_ack(gasnet_node_t node)
{
  gasneti_mutex_lock(&gasnetc_conn_tbl_lock);
  #if !GASNETI_THREADS
    gasneti_assert((GASNETC_NODE2CEP(node))->sq_sema_p == &gasnetc_zero_sema);
  #else
    if (GASNETC_NODE2CEP(node)->sq_sema_p != &gasnetc_zero_sema) {
      /* We are not the first thread to notice the situation */
    } else
  #endif
    {
      gasnetc_conn_t *conn = gasnetc_get_conn(node);

      /* The only valid states are
       * + GASNETC_CONN_STATE_RTU_SENT
       *     The ACK has not yet been received
       * + GASNETC_CONN_STATE_ACK_RCVD
       *     The ACK was received in the same Poll as the current AM Request
       *     and therefore gasnetc_connect_to() has not yet regained control.
       *
       * The !conn case is impossible in the single-threaded case, and in the
       * multi-threaded case is caught by (sq_sema_p != gasnetc_zero_sema) above.
       */
      gasneti_assert(conn && ((conn->state == GASNETC_CONN_STATE_RTU_SENT) ||
                              (conn->state == GASNETC_CONN_STATE_ACK_RCVD)));

      (void) gasnetc_set_sq_sema(&conn->info);
      conn->state = GASNETC_CONN_STATE_DONE;
      gasnetc_dynamic_done(conn, 1);
      GASNETC_STAT_EVENT(CONN_IMPLIED_ACK);
    }
  gasneti_mutex_unlock(&gasnetc_conn_tbl_lock);
}

extern void
gasnetc_conn_rcv_wc(gasnetc_wc_t *comp)
{
  gasnetc_ud_rcv_desc_t *desc = (gasnetc_ud_rcv_desc_t *)(1 ^ (uintptr_t)comp->gasnetc_f_wr_id);
  gasnetc_conn_cmd_t cmd = (gasnetc_conn_cmd_t)(comp->imm_data & 0xff);
  gasnet_node_t node = comp->imm_data >> 16;
  gasneti_tick_t now = gasneti_ticks_now();

#if GASNET_DEBUG /* Drop 1 in N to aid debugging */
  if (gasnetc_conn_drop_denom && !gasnetc_conn_rand_int(gasnetc_conn_drop_denom)) {
    gasnetc_rcv_post_ud(desc);
    return;
  }
#endif

  gasneti_mutex_lock(&gasnetc_conn_tbl_lock);
  {
    gasnetc_conn_t *conn = gasnetc_get_conn(node);
    gasnetc_conn_state_t state = conn ? conn->state : GASNETC_CONN_STATE_DONE;

    /* extract any remote data from the payload and repost desc ASAP */
    if (((state == GASNETC_CONN_STATE_NONE) || (state == GASNETC_CONN_STATE_REQ_SENT)) &&
        ((cmd == GASNETC_CONN_CMD_REQ) || (cmd == GASNETC_CONN_CMD_REP))) {
      gasnetc_conn_info_t *conn_info = &conn->info;
      void *payload = (void *)((uintptr_t)desc->sg.addr + GASNETC_GRH_SIZE);

    #if GASNETC_IBV_XRC
      if (gasnetc_use_xrc) {
        gasnetc_xrc_conn_data_t *data = (gasnetc_xrc_conn_data_t *)payload;
        int qpi;

        for (qpi = 0; qpi < gasnetc_alloc_qps; ++qpi) {
          conn_info->xrc_remote_srq_num[qpi] = data[qpi].srq_num;
          conn_info->remote_xrc_qpn[qpi]     = data[qpi].xrc_qpn;
          conn_info->remote_qpn[qpi]         = data[qpi].qpn;
        }
      } else
    #endif
      {
        memcpy(conn_info->remote_qpn, payload, conn_ud_msg_sz);
      }
    }
    gasnetc_rcv_post_ud(desc);

    /* Now determine the action to take */
    switch (cmd) {
    case GASNETC_CONN_CMD_REQ:
      if (state == GASNETC_CONN_STATE_NONE) {
        /* Normal case */
        (void) gasnetc_qp_create(&conn->info);
        state = GASNETC_CONN_STATE_REP_SENT;
        conn_send_rep(conn);
        conn->reply_time = now;
      } else
      if (state == GASNETC_CONN_STATE_REP_SENT) {
        /* Resend case */
        if (gasneti_ticks_to_ns(now - conn->reply_time) < gasnetc_conn_retransmit_min_ns) {
          /* Recvd impossibly fast, indicating we were inattentive - don't resend yet. */
          GASNETC_STAT_EVENT(CONN_NOREP);
        } else {
          conn_send_rep(conn);
        }
        break; /* Do not advance QP state on resend */
      } else if (state == GASNETC_CONN_STATE_REQ_SENT) {
        /* Resolve the active-active case by picking a winner and a loser. */
        /* Use of odd/even spreads choice uniformly (not biased to high or low nodes) */
        int higher = (node > gasneti_mynode);
        int odd_even = (node ^ gasneti_mynode) & 1;
        if (higher ^ odd_even) {
          state = GASNETC_CONN_STATE_REP_SENT;
          conn->ref_count = 2;
          GASNETC_STAT_EVENT(CONN_AAP);
          /* ...falls through to advance QP... */
        } else {
          state = GASNETC_CONN_STATE_REP_RCVD;
          GASNETC_STAT_EVENT(CONN_AAA);
          break; /* do not advance QP state */
        }
      } else if (state == GASNETC_CONN_STATE_RTU_SENT) {
        /* We must have "won" the active-active race while peer must have missed our REQ */
        conn_send_req(conn);
        break; /* Do not advance QP state */
      } else {
        break; /* Do not advance QP state */
      }

      /* Advance QP state, overlapped w/ network round-trip (if any) and remote work: */
      if (conn->state == GASNETC_CONN_STATE_NONE) {
        (void) gasnetc_qp_reset2init(&conn->info);
      }
      (void) gasnetc_qp_init2rtr(&conn->info);
      gasnetc_sndrcv_attach_peer(node, conn->info.cep);
      (void) gasnetc_qp_rtr2rts(&conn->info);
      (void) gasnetc_set_sq_sema(&conn->info);
      break;

    case GASNETC_CONN_CMD_REP:
      if (state == GASNETC_CONN_STATE_REQ_SENT) {
        /* Normal case */
        state = GASNETC_CONN_STATE_REP_RCVD;
      }
      break;

    case GASNETC_CONN_CMD_RTU:
     {
      /* Since conn is freed after sending the first ACK this "cache" is the
       * best we can do "on the cheap" w/o something like TCP's TIME_WAIT.
       */
      #define GASNETC_ACK_CACHE_SLOTS 8 /* Must be a power of 2 */
      static gasneti_tick_t prev_ack_time[GASNETC_ACK_CACHE_SLOTS] = {0};
      static gasnet_node_t  prev_ack_node[GASNETC_ACK_CACHE_SLOTS] = {0};
      const unsigned int slot = ((unsigned int)node) & (GASNETC_ACK_CACHE_SLOTS - 1);

      if (state == GASNETC_CONN_STATE_REP_SENT) {
        /* Normal case */
        gasneti_sync_writes(); /* "finalize" cep data */
        GASNETC_NODE2CEP(node) = conn->info.cep;
        state = GASNETC_CONN_STATE_DONE;
        gasnetc_dynamic_done(conn, 0);
        conn_send_ack(conn, node);
        prev_ack_time[slot] = now;
        prev_ack_node[slot] = node;
      } else
      if (state == GASNETC_CONN_STATE_DONE) {
        /* Resend case */
        if ((node == prev_ack_node[slot]) &&
            (gasneti_ticks_to_ns(now - prev_ack_time[slot]) < gasnetc_conn_retransmit_min_ns)) {
          /* Recvd impossibly fast, indicating we were inattentive - don't resend yet. */
          GASNETC_STAT_EVENT(CONN_NOACK);
        } else {
          conn_send_ack(NULL, node);
        }
      }
      break;
     }

    case GASNETC_CONN_CMD_ACK:
      if (state == GASNETC_CONN_STATE_RTU_SENT) {
        /* Normal case */
        state = GASNETC_CONN_STATE_ACK_RCVD;
      }
      break;
    }

    if (conn && conn->state != state) {
      conn->state = state;
      if (state == GASNETC_CONN_STATE_DONE) {
        gasnetc_put_conn(conn);
      }
    }
  } while(0);
  gasneti_mutex_unlock(&gasnetc_conn_tbl_lock);
}

extern void
gasnetc_conn_snd_wc(gasnetc_wc_t *comp)
{
  gasnetc_ud_snd_desc_t *desc = (void *)(uintptr_t)(1 ^ comp->gasnetc_f_wr_id);

  gasneti_semaphore_up(conn_ud_hca->snd_cq_sema_p);
  gasnetc_put_ah(desc->ah);
  gasneti_lifo_push(&conn_snd_freelist, desc);
}

#endif /* GASNETC_DYNAMIC_CONNECT */

/* ------------------------------------------------------------------------------------ */
/* Support code for gasnetc_connect() */

/* Convert positive integer to string in base 2 to 36.
 * Returns count of digits actually written, or 0 on overflow.
 * Buffer is '\0' terminated except on overflow.
 * This is the "inverse" to strtol
 */
static int
ltostr(char *buf, int buflen, long val, int base) {
  const char digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
  int i = 0;
  
  gasneti_assert((base > 1) && (base <= strlen(digits)));

  buflen--; /* reserve space for '\0' */
  if_pt (buflen >= 1) {
    /* Work right-aligned in buf */
    char *p = buf + buflen - 1;

    do {
      ldiv_t ld = ldiv(val, base);
      *(p--) = digits[ld.rem];
      val = ld.quot;
    } while ((++i < buflen) && val);
    if_pf (val) return 0; /* Ran out of space */
  
    /* Move to be left-aligned in buf */
    memmove(buf, p+1, i);
    buf[i] = '\0';
  }

  return i;
}

static int
gen_tag(char *tag, int taglen, gasnet_node_t val, int base) {
  int len = ltostr(tag, taglen-1, val, base);
  gasneti_assert(len != 0);
  gasneti_assert(len < taglen-1);
  tag[len+0] = ':';
  tag[len+1] = '\0';
  return len + 1;
}

static long int
my_strtol(const char *ptr, char **endptr, int base) {
  long int result = strtol(ptr,endptr,base);
  if_pf ((*endptr == ptr) || !*endptr) {
    gasneti_fatalerror("Invalid token reading connection table file");
  }
  return result;
}

static gasnet_node_t
get_next_conn(FILE *fp)
{
  static gasnet_node_t range_lo = GASNET_MAXNODES;
  static gasnet_node_t range_hi = 0;

  if (range_lo > range_hi) {
    static char *tok = NULL;

    /* If there is no current token, find the next line with our tag */
    while (!tok || (*tok == '\0')) {
      static char tag[18]; /* even base-2 will fit */
      static size_t taglen = 0;
      if_pf (!taglen) { /* One-time initialization */
        taglen = snprintf(tag, sizeof(tag), "%x:", gasneti_mynode);
      }

      do {
        static int is_header = 1;
        static char *buf = NULL;
        static size_t buflen = 0;
        if (gasneti_getline(&buf, &buflen, fp) == -1) {
          gasneti_free(buf);
          return GASNET_MAXNODES;
        }
        if_pf (is_header) {
          if (!strncmp(buf, "size:", 5)) {
            gasnet_node_t size = my_strtol(buf+5, &tok, 10);
            if (size != gasneti_nodes) {
              gasneti_fatalerror("Connection table input file is for %d nodes rather than %d",
                                 (int)size, (int)gasneti_nodes);
            }
            continue;
          } else if (!strncmp(buf, "base:", 5)) {
            gasnetc_connectfile_in_base = my_strtol(buf+5, &tok, 10);
            taglen = gen_tag(tag, sizeof(tag), gasneti_mynode, gasnetc_connectfile_in_base);
            continue;
          }
          is_header = 0;
        }
        tok = buf;
      } while (strncmp(tok, tag, taglen));
      tok += taglen;
      while (*tok && isspace(*tok)) ++tok;
    }
    
   
    /* Parse the current token */
    { char *p, *q;
      range_lo = range_hi = my_strtol(tok, &p, gasnetc_connectfile_in_base);
      range_hi = (*p == '-') ? my_strtol(p+1, &q, gasnetc_connectfile_in_base) : range_lo;
    }

    /* Advance to next token or end ot line */
    while (*tok && !isspace(*tok)) ++tok;
    while (*tok && isspace(*tok)) ++tok;
  }
   
  return range_lo++;
}
/* ------------------------------------------------------------------------------------ */

/* Setup statically-connected communication */
static int
gasnetc_connect_static(void)
{
  const int             ceps = gasneti_nodes * gasnetc_alloc_qps;
  gasnetc_qpn_t         *local_qpn = gasneti_calloc(ceps, sizeof(gasnetc_qpn_t));
  gasnetc_qpn_t         *remote_qpn = gasneti_calloc(ceps, sizeof(gasnetc_qpn_t));
  gasnetc_conn_info_t   *conn_info = gasneti_calloc(gasneti_nodes, sizeof(gasnetc_conn_info_t));
#if GASNETC_IBV_XRC
  gasnetc_qpn_t         *xrc_remote_rcv_qpn = NULL;
  uint32_t              *xrc_remote_srq_num = NULL;
#endif
  gasnet_node_t         node;
  gasnet_node_t         static_nodes = gasnetc_remote_nodes;
#if GASNETC_IBV_XRC
  gasnet_node_t         static_supernodes = gasneti_nodemap_global_count - 1;
#endif
  int                   i;
  gasnetc_cep_t         *cep; /* First cep of given node */
  uint8_t               *peer_mask = NULL;

  /* Honor user's connections file if given */
  { const char *envstr = gasnetc_connectfile_in;
    if (envstr) {
      const char *filename = gasnetc_parse_filename(envstr);
      FILE *fp;
    #ifdef HAVE_FOPEN64
      fp = fopen64(filename, "r");
    #else
      fp = fopen(filename, "r");
    #endif
      if (!fp) {
        fprintf(stderr, "ERROR: unable to open connection table input file '%s'\n", filename);
      }
      if (filename != envstr) gasneti_free((/* not const */ char *)filename);

      peer_mask = gasneti_calloc(gasneti_nodes, sizeof(uint8_t));
      while (GASNET_MAXNODES != (node = get_next_conn(fp))) {
        gasneti_assert(node < gasneti_nodes);
        peer_mask[node] = 1;
      }
      fclose(fp);

      /* Since conn table is not always symmetric we must transpose-and-OR */
      { uint8_t *transposed_mask = gasneti_malloc(gasneti_nodes * sizeof(uint8_t));
        gasneti_bootstrapAlltoall(peer_mask, sizeof(uint8_t), transposed_mask);
        for (static_nodes = node = 0; node < gasneti_nodes; ++node) {
          peer_mask[node] = !gasnetc_non_ib(node) && (peer_mask[node] || transposed_mask[node]);
          gasneti_assert((peer_mask[node] == 0) || (peer_mask[node] == 1));
          static_nodes += peer_mask[node];
       }
       gasneti_free(transposed_mask);
      }

    #if GASNETC_IBV_XRC
      if (gasnetc_use_xrc) {
        uint8_t *supernode_mask = gasneti_calloc(gasneti_nodemap_global_count, sizeof(uint8_t));
        for (node = 0; node < gasneti_nodes; ++node) {
          supernode_mask[gasneti_nodeinfo[node]] |= peer_mask[node];
        }
        for (static_supernodes = node = 0; node < gasneti_nodemap_global_count; ++node) {
          gasneti_assert((supernode_mask[node] == 0) || (supernode_mask[node] == 1));
          static_supernodes += supernode_mask[node];
        }
        gasneti_free(supernode_mask);
      }
    #endif
    }
  }

  #define GASNETC_IS_REMOTE_NODE(_node) \
    (peer_mask ? peer_mask[_node] : !gasnetc_non_ib(_node))

  #define GASNETC_FOR_EACH_REMOTE_NODE(_node) \
    for ((_node) = 0; (_node) < gasneti_nodes; ++(_node)) \
      if (GASNETC_IS_REMOTE_NODE(_node))

  /* Allocate the dense CEP table and populate the node2cep table. */
  {
    gasnetc_cep_t *cep_table = (gasnetc_cep_t *)
      gasnett_malloc_aligned(GASNETI_CACHE_LINE_BYTES,
                             static_nodes * gasnetc_alloc_qps * sizeof(gasnetc_cep_t));
    for (node = 0, cep = cep_table; node < gasneti_nodes; ++node) { /* NOT randomized */
      if (!GASNETC_IS_REMOTE_NODE(node)) continue;
      gasnetc_node2cep[node] = cep;
      memset(cep, 0, gasnetc_alloc_qps * sizeof(gasnetc_cep_t));
      cep +=  gasnetc_alloc_qps;
    }
    gasneti_assert((cep - cep_table) == (static_nodes * gasnetc_alloc_qps));
  }

  /* Preallocate the SQ semaphores */
#if GASNETC_IBV_XRC
  if (gasnetc_use_xrc) {
    sq_sema_alloc(static_supernodes * gasnetc_alloc_qps);
  } else
#endif
  {
    sq_sema_alloc(static_nodes * gasnetc_alloc_qps);
  }

  /* Initialize connection tracking info and create QPs */
  GASNETC_FOR_EACH_REMOTE_NODE(node) {
    i = node * gasnetc_alloc_qps;
    conn_info[node].node           = node;
    conn_info[node].cep            = GASNETC_NODE2CEP(node);
    conn_info[node].local_qpn      = &local_qpn[i];
  #if GASNETC_IBV_XRC
    conn_info[node].local_xrc_qpn  = &gasnetc_xrc_rcv_qpn[i];
  #endif
    gasnetc_setup_ports(&conn_info[node]);
    (void)gasnetc_qp_create(&conn_info[node]);
  }

  /* exchange address info */
#if GASNETC_IBV_XRC
  if (gasnetc_use_xrc) {
    /* Use single larger exchange rather then multiple smaller ones */
    gasnetc_xrc_conn_data_t *local_tmp  = gasneti_calloc(ceps,  sizeof(gasnetc_xrc_conn_data_t));
    gasnetc_xrc_conn_data_t *remote_tmp = gasneti_malloc(ceps * sizeof(gasnetc_xrc_conn_data_t));
    for (i = node = 0; node < gasneti_nodes; ++node) {
      int qpi;
      cep = GASNETC_NODE2CEP(node);
      for (qpi = 0; qpi < gasnetc_alloc_qps; ++qpi, ++i) {
        if (GASNETC_IS_REMOTE_NODE(node)) {
          gasnetc_hca_t *hca = cep[qpi].hca;
          struct ibv_srq *srq = GASNETC_QPI_IS_REQ(qpi) ? hca->rqst_srq : hca->repl_srq;
          local_tmp[i].srq_num = srq->xrc_srq_num;
        }
        local_tmp[i].xrc_qpn = gasnetc_xrc_rcv_qpn[i];
        local_tmp[i].qpn     = local_qpn[i];
      }
    }
    gasneti_bootstrapAlltoall(local_tmp,
                              gasnetc_alloc_qps * sizeof(gasnetc_xrc_conn_data_t),
                              remote_tmp);
    xrc_remote_rcv_qpn = gasneti_malloc(ceps * sizeof(gasnetc_qpn_t));
    xrc_remote_srq_num = gasneti_malloc(ceps * sizeof(uint32_t));
    for (i = 0; i < ceps; ++i) {
      xrc_remote_srq_num[i] = remote_tmp[i].srq_num;
      xrc_remote_rcv_qpn[i] = remote_tmp[i].xrc_qpn;
      remote_qpn[i]         = remote_tmp[i].qpn;
    }
    gasneti_free(remote_tmp);
    gasneti_free(local_tmp);
  } else
#endif
  gasneti_bootstrapAlltoall(local_qpn, gasnetc_alloc_qps*sizeof(gasnetc_qpn_t), remote_qpn);

  /* Advance state RESET -> INIT -> RTR. */
  GASNETC_FOR_EACH_REMOTE_NODE(node) {
    i = node * gasnetc_alloc_qps;
    conn_info[node].remote_qpn     = &remote_qpn[i];
  #if GASNETC_IBV_XRC
    conn_info[node].remote_xrc_qpn = &xrc_remote_rcv_qpn[i];
    conn_info[node].xrc_remote_srq_num = &xrc_remote_srq_num[i];
  #endif

    (void)gasnetc_qp_reset2init(&conn_info[node]);
    (void)gasnetc_qp_init2rtr(&conn_info[node]);
  }

  /* QPs must reach RTS before we may continue
     (not strictly necessary in practice as long as we don't try to send until peers do.) */
  gasneti_bootstrapBarrier();

  /* Advance state RTR -> RTS */
  GASNETC_FOR_EACH_REMOTE_NODE(node) {
    (void)gasnetc_qp_rtr2rts(&conn_info[node]);
    (void)gasnetc_set_sq_sema(&conn_info[node]);
    GASNETC_STAT_EVENT(CONN_STATIC);
  }

done:
#if GASNETC_IBV_XRC
  gasneti_free(xrc_remote_srq_num);
  gasneti_free(xrc_remote_rcv_qpn);
#endif
  gasneti_free(conn_info);
  gasneti_free(remote_qpn);
  gasneti_free(local_qpn);
  gasneti_free(peer_mask);


  return static_nodes;
} /* gasnetc_connect_static */

/* Setup statically-connected communication and prepare for dynamic connections */
extern int
gasnetc_connect_init(void)
{
  int do_static = 1;
#if GASNETC_DYNAMIC_CONNECT
  int do_dynamic = 0;
#endif
  int fully_connected = 0;

  /* Allocate node->cep lookup table */
  { size_t size = gasneti_nodes*sizeof(gasnetc_cep_t *);
    gasnetc_node2cep = (gasnetc_cep_t **)
      gasnett_malloc_aligned(GASNETI_CACHE_LINE_BYTES, size);
    memset(gasnetc_node2cep, 0, size);
  }

  if_pf (!gasnetc_remote_nodes) {
    GASNETI_TRACE_PRINTF(I, ("No connection setup since there are no remote nodes"));
    return GASNET_OK;
  }

#if GASNETC_DYNAMIC_CONNECT
  /* Parse connection related env vars */
 #if GASNET_DEBUG
  gasnetc_conn_drop_denom =
        gasneti_getenv_int_withdefault("GASNET_CONNECT_DROP_DENOM", 0, 0);
 #endif
  gasnetc_connectfile_in  = gasnet_getenv("GASNET_CONNECTFILE_IN");
  if (gasnetc_connectfile_in && !gasnetc_connectfile_in[0]) { /* empty string */
    gasnetc_connectfile_in = NULL;
  }
  gasnetc_connectfile_out = gasnet_getenv("GASNET_CONNECTFILE_OUT");
  if (gasnetc_connectfile_out && !gasnetc_connectfile_out[0]) { /* empty string */
    gasnetc_connectfile_out = NULL;
  }
  gasnetc_connectfile_out_base =
        gasneti_getenv_int_withdefault("GASNET_CONNECTFILE_BASE",
                                       gasnetc_connectfile_out_base, 0);

  do_static = gasneti_getenv_int_withdefault("GASNET_CONNECT_STATIC", 1, 0);
  do_dynamic = gasneti_getenv_int_withdefault("GASNET_CONNECT_DYNAMIC", 1, 0);
  if (!do_static && !do_dynamic) {
    if (!gasneti_mynode) {
      fprintf(stderr, "WARNING: Both GASNET_CONNECT_STATIC and GASNET_CONNECT_DYNAMIC are non-zero.\n"
                      "         Enabling dynamic connection support.\n");
    }
    do_dynamic = 1;
  }

  { /* Env vars are in us, but internal vars are in ns */
    int64_t tmp_min, tmp_max;
    tmp_min = gasnetc_conn_retransmit_min_ns / 1000;
    tmp_min = gasneti_getenv_int_withdefault("GASNET_CONNECT_RETRANS_MIN", tmp_min, 0);
    tmp_max = gasnetc_conn_retransmit_max_ns / 1000;
    tmp_max = gasneti_getenv_int_withdefault("GASNET_CONNECT_RETRANS_MAX", tmp_max, 0);

    if (tmp_min >= tmp_max) {
      if (!gasneti_mynode) {
        fprintf(stderr, "WARNING: GASNET_CONNECT_RETRANS_MIN >= GASNET_CONNECT_RETRANS_MAX.  "
                        "Using default values instead.\n");
      }
    } else {
      gasnetc_conn_retransmit_min_ns = tmp_min * 1000;
      gasnetc_conn_retransmit_max_ns = tmp_max * 1000;
    }
  }
#endif

  /* Determine the inline data limit given the QP parameters we will use. */
  {
    const size_t orig_inline_limit = gasnetc_inline_limit;
    int i;
 
    for (i = 0; i < gasnetc_num_ports; ++i) {
      gasnetc_check_inline_limit(i, gasnetc_op_oust_pp, GASNETC_SND_SG);
      if (gasnetc_use_srq) {
        /* Corresponds to a Request QP */
        gasnetc_check_inline_limit(i, gasnetc_am_oust_pp, 1);
      }
    }

    /* warn on reduced inline limit */
    if ((orig_inline_limit != (size_t)-1) && (gasnetc_inline_limit < orig_inline_limit)) {
    #if GASNET_CONDUIT_IBV
      if (gasnet_getenv("GASNET_INLINESEND_LIMIT") != NULL)
    #endif
        fprintf(stderr,
                "WARNING: Requested GASNET_INLINESEND_LIMIT %d reduced to HCA limit %d\n",
                (int)orig_inline_limit, (int)gasnetc_inline_limit);
    }

    GASNETI_TRACE_PRINTF(I, ("Final/effective GASNET_INLINESEND_LIMIT = %d", (int)gasnetc_inline_limit));
    gasnetc_sndrcv_init_inline();
  }

  /* Create static connections unless disabled */
  if (do_static) {
    gasnet_node_t static_nodes = gasnetc_connect_static();
    fully_connected = (static_nodes == gasnetc_remote_nodes);
    GASNETI_TRACE_PRINTF(I, ("%s connected at startup to %d of %d remote nodes",
                             fully_connected ? "Fully" : "Partially",
                             (int)static_nodes, (int)gasnetc_remote_nodes));
  } else {
    GASNETI_TRACE_PRINTF(I, ("Static connection at startup has been disabled at user request"));
  }

#if GASNETC_DYNAMIC_CONNECT
  if (!do_dynamic) {
    GASNETI_TRACE_PRINTF(I, ("Dynamic connection has been disabled at user request"));
  } else if (do_static && !gasnetc_connectfile_in) {
    GASNETI_TRACE_PRINTF(I, ("Dynamic connection automatically disabled for fully-connected job"));
  } else {
    /* TODO: allow env var to select specific port for UD */
    gasnetc_qp_setup_ud(&gasnetc_port_tbl[0], fully_connected);
  }
#else
  GASNETI_TRACE_PRINTF(I, ("Dynamic connection was disabled at library build time"));
#endif

  return GASNET_OK;
} /* gasnetc_connect_init */

/* ------------------------------------------------------------------------------------ */
/* Support code for gasneti_conn_fini */

static char dump_conn_line[512] = "";
static gasnet_node_t dump_conn_first = GASNET_MAXNODES;
static gasnet_node_t dump_conn_prev;

static void
dump_conn_write(int fd, const char *buf, size_t len)
{
  /* TODO: loop w/ retry on whort writes? */
  ssize_t rc = write(fd, buf, len);
  if_pf (rc != len) {
    gasneti_fatalerror("Write to connection file failed or truncated: rc=%ld errno=%s(%i)",
                       (long int)rc, strerror(errno), errno);
  }
}

static void
dump_conn_outln(int fd)
{
  static char fullline[96];
  static size_t taglen = 0;
  size_t len;

  if_pf (!taglen) {
    taglen = gen_tag(fullline, sizeof(fullline), gasneti_mynode, gasnetc_connectfile_out_base);
  }

  len = strlen(dump_conn_line+1);
  memcpy(fullline+taglen, dump_conn_line+1, len);

  len += taglen;
  fullline[len] = '\n';

  dump_conn_write(fd, fullline, len+1);
}

static void
dump_conn_out(int fd) {
  char elem[35];
  size_t len, tmp;

  /* Leading space */
  elem[0] = ' ';
  len = 1;

  /* dump_conn_first foramtted with desired base */
  tmp = ltostr(elem+len, sizeof(elem)-len, dump_conn_first, gasnetc_connectfile_out_base);
  gasneti_assert(tmp != 0);
  len += tmp;

  if (dump_conn_prev != dump_conn_first) {
    /* Choose ' ' or '-' as separator */
    elem[len++] = ((dump_conn_prev - dump_conn_first) > 1) ? '-' : ' ';
  
    /* dump_conn_prev foramtted with desired base */
    tmp = ltostr(elem+len, sizeof(elem)-len, dump_conn_prev, gasnetc_connectfile_out_base);
    gasneti_assert(tmp != 0);
    len += tmp;

    elem[len] = '\0';
  }

  gasneti_assert(len == strlen(elem));
  gasneti_assert(len < sizeof(elem));

  if (strlen(dump_conn_line) + len < sizeof(dump_conn_line)) {
    strcat(dump_conn_line, elem);
  } else {
    dump_conn_outln(fd);
    strcpy(dump_conn_line, elem);
  }
}

static void
dump_conn_next(int fd, gasnet_node_t n)
{
  if (dump_conn_first == GASNET_MAXNODES) {
    dump_conn_first = dump_conn_prev = n;
    return;
  } else if (n == dump_conn_prev+1) {
    dump_conn_prev = n;
    return;
  }

  dump_conn_out(fd);
  dump_conn_first = dump_conn_prev = n;
}

static void
dump_conn_done(int fd)
{
  if (dump_conn_first == GASNET_MAXNODES) return;
  dump_conn_out(fd);
  dump_conn_outln(fd);
  close(fd);
}

/* ------------------------------------------------------------------------------------ */

/* Fini currently just dumps connection table. */
extern int
gasnetc_connect_fini(void)
{
  gasnet_node_t n, count = 0;
  int fd = -1;

  /* Open file replacing any '%' in filename with node number */
  { const char *envstr = gasnetc_connectfile_out;
    if (envstr) {
      const char *filename = gasnetc_parse_filename(envstr);
      int flags = O_APPEND | O_CREAT | O_WRONLY;
    #ifdef O_LARGEFILE
      flags |= O_LARGEFILE;
    #endif
      fd = open(filename, flags, S_IRUSR | S_IWUSR);
      if (fd < 0) {
        fprintf(stderr, "ERROR: unable to open connection table output file '%s'\n", filename);
      }
      if (filename != envstr) gasneti_free((/* not const */ char *)filename);
      if (!gasneti_mynode || strchr(envstr, '%')) {
        char buf[16];
        size_t len;
        int rc = ftruncate(fd,0);
        if_pf (rc < 0) {
            gasneti_fatalerror("Failed to truncate connection file: rc=%d errno=%s(%i)",
                               rc, strerror(errno), errno);
        }
        len = snprintf(buf, sizeof(buf), "size:%d\n", gasneti_nodes);
        dump_conn_write(fd, buf, len);
        len = snprintf(buf, sizeof(buf), "base:%d\n", gasnetc_connectfile_out_base);
        dump_conn_write(fd, buf, len);
      }
      gasneti_bootstrapBarrier();
    }
  }
 
  for (n = 0; n < gasneti_nodes; ++n) {
    gasnetc_cep_t *cep = GASNETC_NODE2CEP(n);
    int qpi;
    if (!cep) continue;
    for (qpi=0; qpi<gasnetc_alloc_qps; ++qpi, ++cep) {
      if (cep[qpi].used) {
        if (fd >= 0) dump_conn_next(fd, n);
        ++count;
        break;
      }
    }
  }
  if (fd >= 0) dump_conn_done(fd);
  GASNETI_TRACE_PRINTF(C, ("Network traffic sent to %d of %d remote nodes", (int)count, (int)gasnetc_remote_nodes));

  return GASNET_OK;
} /* gasnetc_connect_fini */
