document.addEventListener("DOMContentLoaded", function() {
	const li = document.querySelectorAll(".drawer__list-item");
	li.forEach((el, index) => {
		el.addEventListener("click", () => {
			console.log("Clicked!", index);
		});
	});

	const processing = document.querySelectorAll(".processing");
	console.log(processing);

	if (processing.length > 0) {
		processing.forEach(p => {
			p.addEventListener("click", () => {
				console.log("Clicked!", p.dataset.id);
				fetch(`http://localhost:5000/${p.dataset.url}?_=` + new Date().getTime(), {
					method: "POST", // or 'PUT'
					headers: {
						"Content-Type": "application/json"
						// "Cache-Control": "no-cache, must-revalidate"
					},
					body: `{ "mt": "${p.dataset.id}" }`
				})
					.then(res => {
						return res.json();
					})
					.then(res => {
						const image = `<img src='${res + "?" + new Date().getTime()}' alt='filtered image' />`;
						const imageDiv = document.querySelector("#image");
						imageDiv.innerHTML = image;
					})
					.catch(err => {
						console.log(err);
					});
			});
		});
	}
});


