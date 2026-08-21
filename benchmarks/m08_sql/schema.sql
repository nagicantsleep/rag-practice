CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT NOT NULL, region TEXT NOT NULL, segment TEXT NOT NULL, active INTEGER NOT NULL);
CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, unit_price REAL NOT NULL);
CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL, order_date TEXT NOT NULL, status TEXT NOT NULL, FOREIGN KEY(customer_id) REFERENCES customers(id));
CREATE TABLE order_items(order_id INTEGER NOT NULL, product_id INTEGER NOT NULL, quantity INTEGER NOT NULL, unit_price REAL NOT NULL, PRIMARY KEY(order_id, product_id), FOREIGN KEY(order_id) REFERENCES orders(id), FOREIGN KEY(product_id) REFERENCES products(id));
