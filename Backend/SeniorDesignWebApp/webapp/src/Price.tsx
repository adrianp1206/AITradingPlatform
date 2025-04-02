import * as React from "react";
import { useParams, useLoaderData, Link } from "react-router-dom";

export const Price: React.FC = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const price = useLoaderData() as number;

  const formattedPrice = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(price);

  return (
    <div>
      <h2>Price for {symbol}</h2>
      <p>{formattedPrice}</p>

      <Link to="/symbols">Back to Symbols</Link>
    </div>
  );
};
