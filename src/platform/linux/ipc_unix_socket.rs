use interprocess;
use interprocess::os::unix::udsocket::{self, UdSocket, UdStream, UdStreamListener};
use std::cmp::min;
use std::io::{Error, ErrorKind, IoSlice, IoSliceMut};
use std::mem::size_of;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::thread;
use std::time::{Duration, SystemTime};
use vulkano::VulkanObject;

use super::ipc_commands::{CommandMsg, ResultMsg};

struct IPCConnection {
    conn: UdStream,
    proc_id: i32,
    timeout: Duration,
}

struct IPCSocket {
    listener_socket: UdStreamListener,
    connections: Vec<IPCConnection>,
    timeout: Duration,
}

impl IPCConnection {
    pub fn new(conn: UdStream, proc_id: i32, timeout: Duration) -> IPCConnection {
        conn.set_nonblocking(true).unwrap();
        IPCConnection {
            conn,
            proc_id,
            timeout,
        }
    }

    pub fn try_connect(
        socket_path: &str,
        timeout: Duration,
    ) -> Result<Option<IPCConnection>, Error> {
        IPCConnection::try_fcn_timeout(
            || match UdStream::connect(socket_path) {
                Ok(c) => {
                    let pid = 0; //c.get_peer_credentials().unwrap().pid;
                    Ok(Some(IPCConnection::new(c, pid, timeout)))
                }
                Err(e) => match e.kind() {
                    ErrorKind::AddrNotAvailable | ErrorKind::NotFound | ErrorKind::Interrupted => {
                        Ok(None)
                    }
                    _ => Err(e),
                },
            },
            &timeout,
            &IPCConnection::sleep_interval(timeout),
        )
    }

    pub fn send_anillary_handles(&self, handles: &[RawFd]) -> Result<usize, Error> {
        if handles.len() == 0 {
            Ok(0)
        } else {
            let ancillary =
                [AncillaryData::FileDescriptors(std::borrow::Cow::Borrowed(handles)); 1];
            self.conn
                .send_ancillary(&[1 as u8; 4], ancillary)
                .map(|r| r.1)
        }
    }

    pub fn recv_ancillary(&self, handle_count: usize) -> Result<Vec<OwnedFd>, Error> {
        let mut buf = [0 as u8; 4];
        let mut fds = Vec::<OwnedFd>::new();
        let _ = IPCConnection::try_fcn_timeout(
            || {
                let mut abuf = AncillaryDataBuf::Owned(vec![
                        0 as u8;
                        AncillaryData::encoded_size_of_file_descriptors(
                            handle_count// - fds.len()
                        )
                    ]);
                let rec = self.conn.recv_ancillary(&mut buf, &mut abuf)?.1;
                if rec > 0 {
                    for adat in abuf.decode().into_iter() {
                        if let AncillaryData::FileDescriptors(rfds) = adat {
                            for fd in rfds.into_iter() {
                                fds.push(unsafe { OwnedFd::from_raw_fd(*fd) });
                            }
                        }
                    }

                    if fds.len() >= handle_count {
                        Ok(Some(fds.len()))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            },
            &self.timeout,
            &IPCConnection::sleep_interval(self.timeout),
        )?;

        Ok(fds)
    }

    pub fn send_command(&self, command_msg: CommandMsg) -> Result<usize, Error> {
        const MSG_LEN: usize = size_of::<CommandMsg>();
        let raw_ptr: *const CommandMsg = &(command_msg);
        let msg: &[u8; MSG_LEN] = unsafe { raw_ptr.cast::<[u8; MSG_LEN]>().as_ref().unwrap() };
        self.conn.send(msg)
    }

    pub fn send_result(&self, result_msg: ResultMsg) -> Result<usize, Error> {
        const MSG_LEN: usize = size_of::<ResultMsg>();
        let raw_ptr: *const ResultMsg = &(result_msg);
        let msg: &[u8; MSG_LEN] = unsafe { raw_ptr.cast::<[u8; MSG_LEN]>().as_ref().unwrap() };
        self.conn.send(msg)
    }

    pub fn recv_command(&self) -> Result<Option<CommandMsg>, Error> {
        let mut msg = CommandMsg::default();

        const MSG_LEN: usize = size_of::<CommandMsg>();
        let buf: &mut [u8; MSG_LEN] = unsafe {
            (&mut msg as *mut CommandMsg)
                .cast::<[u8; MSG_LEN]>()
                .as_mut()
                .unwrap()
        };

        let mut rec_bytes: usize = 0;
        let recv_res = IPCConnection::try_fcn_timeout(
            || {
                let rec_buf: &mut [u8] = buf.split_at_mut(rec_bytes).1;
                rec_buf[1] = 100;
                rec_bytes += self.conn.recv(rec_buf)?;

                if rec_bytes >= MSG_LEN {
                    Ok::<Option<()>, Error>(Some(()))
                } else {
                    Ok(None)
                }
            },
            &self.timeout,
            &IPCConnection::sleep_interval(self.timeout),
        )?;

        Ok(recv_res.and_then(|_| Some(msg)))
    }

    pub fn recv_result(&self) -> Result<Option<ResultMsg>, Error> {
        let mut msg = ResultMsg::default();

        const MSG_LEN: usize = size_of::<ResultMsg>();
        let buf: &mut [u8; MSG_LEN] = unsafe {
            (&mut msg as *mut ResultMsg)
                .cast::<[u8; MSG_LEN]>()
                .as_mut()
                .unwrap()
        };

        let mut rec_bytes: usize = 0;
        let recv_res = IPCConnection::try_fcn_timeout(
            || {
                let rec_buf = buf.split_at_mut(rec_bytes).1;
                rec_bytes += self.conn.recv(rec_buf)?;

                if rec_bytes >= MSG_LEN {
                    Ok::<Option<()>, Error>(Some(()))
                } else {
                    Ok(None)
                }
            },
            &self.timeout,
            &IPCConnection::sleep_interval(self.timeout),
        )?;

        Ok(recv_res.and_then(|_| Some(msg)))
    }

    pub fn send_ack(&self) -> Result<(), Error> {
        self.conn.send(&[1 as u8]).map(|_| ())
    }

    pub fn recv_ack(&self) -> Result<Option<()>, Error> {
        let mut buf = [0 as u8];
        IPCConnection::try_fcn_timeout(
            || {
                self.conn.recv(&mut buf).map(|read_count| match read_count {
                    0 => None,
                    _ => Some(()),
                })
            },
            &self.timeout,
            &IPCConnection::sleep_interval(self.timeout),
        )
    }

    fn try_fcn_timeout<R, F: FnMut() -> Result<Option<R>, Error>>(
        mut f: F,
        timeout: &Duration,
        sleep_time: &Duration,
    ) -> Result<Option<R>, Error> {
        let start_time = SystemTime::now();
        loop {
            let r = match f() {
                Err(e) => match e.kind() {
                    ErrorKind::WouldBlock => None,
                    _ => Err(e)?,
                },
                Ok(r) => r,
            };

            if r.is_some() {
                break Ok(r);
            }

            if SystemTime::now().duration_since(start_time).unwrap() > *timeout {
                break Ok(None);
            }

            thread::sleep(*sleep_time);
        }
    }

    fn sleep_interval(timeout: Duration) -> Duration {
        const MIN_SLEEP_DUR: Duration = Duration::from_millis(100);
        min(MIN_SLEEP_DUR, timeout / 10)
    }
}

impl IPCSocket {
    pub fn new(socket_path: &str, timeout: Duration) -> Result<IPCSocket, Error> {
        let listener_socket = IPCConnection::try_fcn_timeout(
            || match UdStreamListener::bind_with_drop_guard(socket_path) {
                Err(e) => match e.kind() {
                    ErrorKind::AddrInUse
                    | ErrorKind::AddrNotAvailable
                    | ErrorKind::AlreadyExists => Ok(None),
                    _ => Err(e),
                },
                Ok(r) => Ok(Some(r)),
            },
            &timeout,
            &IPCConnection::sleep_interval(timeout),
        )?
        .expect("Failed to create socket");
        listener_socket.set_nonblocking(true)?;

        Ok(IPCSocket {
            listener_socket,
            connections: Vec::new(),
            timeout,
        })
    }

    pub fn try_accept(&mut self) -> Result<Option<&IPCConnection>, Error> {
        let res = IPCConnection::try_fcn_timeout(
            || {
                print!("Trying to accept\n");
                match self.listener_socket.accept() {
                    Err(e) => match e.kind() {
                        ErrorKind::WouldBlock => Ok(None),
                        _ => Err(e),
                    },
                    Ok(c) => {
                        let pid = 0; //c.get_peer_credentials().unwrap().pid;
                        let ipc_conn = IPCConnection::new(c, pid, self.timeout);
                        self.connections.push(ipc_conn);
                        Ok(Some(()))
                    }
                }
            },
            &self.timeout,
            &IPCConnection::sleep_interval(self.timeout),
        )?;

        if res.is_some() {
            Ok(Some(self.connections.last().unwrap()))
        } else {
            Ok(None)
        }
    }
}

//#[cfg(tests)]
mod tests {
    use std::{fs, mem::size_of_val, os::fd::AsRawFd};

    use libc::{c_int, SOL_SOCKET, SO_PASSCRED};

    use super::*;

    const TIMEOUT: Duration = Duration::from_millis(10000);
    const SOCK_PATH: &str = "test_socket.sock";

    #[test]
    fn socket_creation() {
        let _ = IPCSocket::new(SOCK_PATH, TIMEOUT).unwrap();
    }

    fn _ipc_stream_create() -> (IPCSocket, IPCConnection) {
        let _ = fs::remove_file(SOCK_PATH);

        let listen_thread = move || {
            let mut listener = IPCSocket::new(SOCK_PATH, TIMEOUT)?;
            let _ = listener
                .try_accept()?
                .expect("Failed to listen for connection");
            Ok::<_, Error>(listener)
        };

        let connect_thread = || {
            let mut conn = IPCConnection::try_connect(SOCK_PATH, TIMEOUT)?
                .expect("Failed to connect to socket");
            Ok::<_, Error>(conn)
        };

        let listen_handle = thread::spawn(listen_thread);
        let connect_handle = thread::spawn(connect_thread);
        let listener = listen_handle.join().unwrap().unwrap();
        let conn = connect_handle.join().unwrap().unwrap();

        (listener, conn)
    }

    #[test]
    fn ipc_stream_create() {
        let _ = _ipc_stream_create();
    }

    #[test]
    fn ipc_ack() {
        let (listener, conn) = _ipc_stream_create();

        let send_thread = move || {
            let send_conn = listener.connections.last().unwrap();
            send_conn.send_ack()
        };

        let recv_thread = move || conn.recv_ack();

        let s_handle = thread::spawn(send_thread);
        let r_handle = thread::spawn(recv_thread);

        let _ = s_handle.join().unwrap().expect("Failed to send ack");
        let _ = r_handle
            .join()
            .unwrap()
            .unwrap()
            .expect("Failed to receive ack");
    }

    #[test]
    fn ipc_msg() {
        let (listener, conn) = _ipc_stream_create();

        let send_thread = move || {
            let send_conn = listener.connections.last().unwrap();
            let mut msg = CommandMsg::default();
            unsafe { (*msg.data.find_img).image_name.fill(1) };
            send_conn.send_command(msg)
        };

        let recv_thread = move || conn.recv_command();

        let s_handle = thread::spawn(send_thread);
        let r_handle = thread::spawn(recv_thread);

        let s_res = s_handle.join().unwrap().expect("Failed to send cmd");
        let r_res = r_handle.join().unwrap().expect("Failed to recv cmd");

        assert_eq!(s_res, size_of::<CommandMsg>());
        assert!(r_res.is_some());

        let mut cmp_msg = CommandMsg::default();
        unsafe { (*cmp_msg.data.find_img).image_name.fill(1) };
        let rec_msg = r_res.unwrap();

        assert_eq!(cmp_msg.tag, rec_msg.tag);
        assert_eq!(unsafe { cmp_msg.data.find_img.image_name }, unsafe {
            rec_msg.data.find_img.image_name
        });
    }

    #[test]
    fn ipc_result_msg() {
        let (listener, conn) = _ipc_stream_create();

        let send_thread = move || {
            let send_conn = listener.connections.last().unwrap();
            let mut msg = ResultMsg::default();
            unsafe { (*msg.data.find_img).img_data.height = 1024 };
            send_conn.send_result(msg)
        };

        let recv_thread = move || conn.recv_result();

        let s_handle = thread::spawn(send_thread);
        let r_handle = thread::spawn(recv_thread);

        let s_res = s_handle.join().unwrap().expect("Failed to send res");
        let r_res = r_handle.join().unwrap().expect("Failed to recv res");

        assert_eq!(s_res, size_of::<ResultMsg>());
        assert!(r_res.is_some());

        let mut cmp_msg = ResultMsg::default();
        unsafe { (*cmp_msg.data.find_img).img_data.height = 1024 };
        let rec_msg = r_res.unwrap();

        assert_eq!(cmp_msg.tag, rec_msg.tag);
        assert_eq!(unsafe { cmp_msg.data.find_img.img_data.height }, unsafe {
            rec_msg.data.find_img.img_data.height
        });
    }

    #[test]
    fn ipc_ancillary() {
        let (listener, conn) = _ipc_stream_create();

        let passcred: libc::c_int = 0;
        unsafe {
            libc::setsockopt(
                conn.conn.as_raw_fd(),
                SOL_SOCKET,
                SO_PASSCRED,
                &passcred as *const _ as *const _,
                size_of_val(&passcred) as u32,
            );

            libc::setsockopt(
                listener.connections.last().unwrap().conn.as_raw_fd(),
                SOL_SOCKET,
                SO_PASSCRED,
                &passcred as *const _ as *const _,
                size_of_val(&passcred) as u32,
            );
        };

        let tmp1 = fs::File::create("test1.txt").unwrap();
        let tmp2 = fs::File::create("test2.txt").unwrap();
        //let tmp1 = tempfile::NamedTempFile::new().unwrap();
        //let tmp2 = tempfile::NamedTempFile::new().unwrap();

        let fd1 = tmp1.as_raw_fd();
        let fd2 = tmp2.as_raw_fd();

        let send_thread = move || {
            let send_conn = listener.connections.last().unwrap();
            let handles = [fd1.as_raw_fd(), fd2.as_raw_fd()];
            send_conn.send_anillary_handles(&handles)
        };

        let recv_thread = move || {
            //thread::sleep(Duration::from_secs(1));
            conn.recv_ancillary(2)
        };

        let s_handle = thread::spawn(send_thread);
        let r_handle = thread::spawn(recv_thread);

        thread::sleep(Duration::from_secs(5));

        let mut r_res = r_handle.join().unwrap().expect("Failed to recv ancillary");
        //let s_res = s_handle.join().unwrap().expect("Failed to send ancillary");

        //assert_ne!(s_res, 0);
        assert_eq!(r_res.len(), 2);
        r_res.clear();

        //tmp1.close().unwrap();
    }
}
