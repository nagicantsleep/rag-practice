INSERT INTO customers VALUES
(1,'Acme Corp','North','enterprise',1),
(2,'Beta LLC','South','smb',1),
(3,'Cora Labs','North','enterprise',1),
(4,'Delta Shop','West','smb',0);
INSERT INTO products VALUES
(101,'VectorDB Pro','database',120.0),
(102,'EmbedKit','ml',80.0),
(103,'QueryGuard','security',50.0),
(104,'ArchiveBox','storage',40.0);
INSERT INTO orders VALUES
(1001,1,'2026-01-10','shipped'),
(1002,2,'2026-02-12','shipped'),
(1003,1,'2026-03-01','cancelled'),
(1004,3,'2026-04-15','shipped'),
(1005,4,'2025-12-20','shipped');
INSERT INTO order_items VALUES
(1001,101,2,120.0),
(1001,103,1,50.0),
(1002,102,3,80.0),
(1003,104,10,40.0),
(1004,101,1,120.0),
(1004,102,1,80.0),
(1005,104,2,40.0);
